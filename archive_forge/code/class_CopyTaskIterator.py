from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import manifest_util
from googlecloudsdk.command_lib.storage import path_util
from googlecloudsdk.command_lib.storage import plurality_checkable_iterator
from googlecloudsdk.command_lib.storage import posix_util
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.command_lib.storage.tasks.cp import copy_task_factory
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
class CopyTaskIterator:
    """Iterates over each expanded source and creates an appropriate copy task."""

    def __init__(self, source_name_iterator, destination_string, custom_md5_digest=None, delete_source=False, do_not_decompress=False, force_daisy_chain=False, print_created_message=False, shared_stream=None, skip_unsupported=True, task_status_queue=None, user_request_args=None):
        """Initializes a CopyTaskIterator instance.

    Args:
      source_name_iterator (name_expansion.NameExpansionIterator):
        yields resource_reference.Resource objects with expanded source URLs.
      destination_string (str): The copy destination path or url.
      custom_md5_digest (str|None): User-added MD5 hash output to send to server
        for validating a single resource upload.
      delete_source (bool): If copy completes successfully, delete the source
        object afterwards.
      do_not_decompress (bool): Prevents automatically decompressing
        downloaded gzips.
      force_daisy_chain (bool): If True, yields daisy chain copy tasks in place
        of intra-cloud copy tasks.
      print_created_message (bool): Print the versioned URL of each successfully
        copied object.
      shared_stream (stream): Multiple tasks may reuse a read or write stream.
      skip_unsupported (bool): Skip creating copy tasks for unsupported object
        types.
      task_status_queue (multiprocessing.Queue|None): Used for estimating total
        workload from this iterator.
      user_request_args (UserRequestArgs|None): Values for RequestConfig.
    """
        self._all_versions = source_name_iterator.object_state is cloud_api.ObjectState.LIVE_AND_NONCURRENT
        self._has_multiple_top_level_sources = source_name_iterator.has_multiple_top_level_resources
        self._has_cloud_source = False
        self._has_local_source = False
        self._source_name_iterator = plurality_checkable_iterator.PluralityCheckableIterator(source_name_iterator)
        self._multiple_sources = self._source_name_iterator.is_plural()
        self._custom_md5_digest = custom_md5_digest
        self._delete_source = delete_source
        self._do_not_decompress = do_not_decompress
        self._force_daisy_chain = force_daisy_chain
        self._print_created_message = print_created_message
        self._shared_stream = shared_stream
        self._skip_unsupported = skip_unsupported
        self._task_status_queue = task_status_queue
        self._user_request_args = user_request_args
        self._total_file_count = 0
        self._total_size = 0
        self._raw_destination = _get_raw_destination(destination_string)
        if self._multiple_sources:
            self._raise_if_destination_is_file_url_and_not_a_directory_or_pipe()
        else:
            self._raise_if_download_destination_ends_with_delimiter_and_does_not_exist()
        if self._multiple_sources and self._custom_md5_digest:
            raise errors.Error('Received multiple objects to upload, but only one custom MD5 digest is allowed.')
        self._already_completed_sources = manifest_util.parse_for_completed_sources(getattr(user_request_args, 'manifest_path', None))

    def _raise_error_if_source_matches_destination(self):
        if not self._multiple_sources and (not self._source_name_iterator.is_empty()):
            source_url = self._source_name_iterator.peek().expanded_url
            if source_url == self._raw_destination.storage_url:
                raise errors.InvalidUrlError('Source URL matches destination URL: {}'.format(source_url))

    def _raise_if_destination_is_file_url_and_not_a_directory_or_pipe(self):
        if isinstance(self._raw_destination.storage_url, storage_url.FileUrl) and (not (_destination_is_container(self._raw_destination) or self._raw_destination.storage_url.is_stream)):
            raise errors.InvalidUrlError('Destination URL must name an existing directory. Provided: {}.'.format(self._raw_destination.storage_url.object_name))

    def _raise_if_download_destination_ends_with_delimiter_and_does_not_exist(self):
        if isinstance(self._raw_destination.storage_url, storage_url.FileUrl):
            destination_path = self._raw_destination.storage_url.object_name
            if destination_path.endswith(self._raw_destination.storage_url.delimiter) and (not self._raw_destination.storage_url.isdir()):
                raise errors.InvalidUrlError('Destination URL must name an existing directory if it ends with a delimiter. Provided: {}.'.format(destination_path))

    def _update_workload_estimation(self, resource):
        """Updates total_file_count and total_size.

    Args:
      resource (resource_reference.Resource): Any type of resource. Parse to
        help estimate total workload.
    """
        if self._total_file_count == -1 or self._total_size == -1:
            return
        try:
            if resource.is_container():
                return
            size = resource.size
            if isinstance(resource, resource_reference.FileObjectResource):
                self._has_local_source = True
            elif isinstance(resource, resource_reference.ObjectResource):
                self._has_cloud_source = True
            else:
                raise errors.ValueCannotBeDeterminedError
        except (OSError, errors.ValueCannotBeDeterminedError):
            if not _resource_is_stream(resource):
                log.error('Could not get size of resource {}.'.format(resource))
            self._total_file_count = -1
            self._total_size = -1
        else:
            self._total_file_count += 1
            self._total_size += size or 0

    def _print_skip_and_maybe_send_to_manifest(self, message, source):
        """Prints why task is being skipped and maybe records in manifest."""
        log.status.Print(message)
        if self._user_request_args and self._user_request_args.manifest_path and self._task_status_queue:
            manifest_util.send_skip_message(self._task_status_queue, source.resource, self._raw_destination, message)

    def __iter__(self):
        self._raise_error_if_source_matches_destination()
        for source in self._source_name_iterator:
            if self._delete_source:
                copy_util.raise_if_mv_early_deletion_fee_applies(source.resource)
            if self._skip_unsupported:
                unsupported_type = resource_util.get_unsupported_object_type(source.resource)
                if unsupported_type:
                    message = resource_util.UNSUPPORTED_OBJECT_WARNING_FORMAT.format(source.resource.storage_url, unsupported_type.value)
                    self._print_skip_and_maybe_send_to_manifest(message, source)
                    continue
            if source.resource.storage_url.url_string in self._already_completed_sources:
                message = 'Skipping item {} because manifest marks it as skipped or completed.'.format(source.resource.storage_url)
                self._print_skip_and_maybe_send_to_manifest(message, source)
                continue
            destination_resource = self._get_copy_destination(self._raw_destination, source)
            source_url = source.resource.storage_url
            destination_url = destination_resource.storage_url
            posix_util.run_if_setting_posix(posix_to_set=None, user_request_args=self._user_request_args, function=posix_util.raise_if_source_and_destination_not_valid_for_preserve_posix, source_url=source_url, destination_url=destination_url)
            if isinstance(source.resource, resource_reference.ObjectResource) and isinstance(destination_url, storage_url.FileUrl) and destination_url.object_name.endswith(destination_url.delimiter):
                log.debug('Skipping downloading {} to {} since the destination ends in a file system delimiter.'.format(source_url.versionless_url_string, destination_url.versionless_url_string))
                continue
            if not self._multiple_sources and source_url.versionless_url_string != source.expanded_url.versionless_url_string:
                self._raise_if_destination_is_file_url_and_not_a_directory_or_pipe()
            if self._custom_md5_digest:
                source.resource.md5_hash = self._custom_md5_digest
            self._update_workload_estimation(source.resource)
            yield copy_task_factory.get_copy_task(source.resource, destination_resource, do_not_decompress=self._do_not_decompress, delete_source=self._delete_source, force_daisy_chain=self._force_daisy_chain, print_created_message=self._print_created_message, print_source_version=source.original_url.generation or self._all_versions, shared_stream=self._shared_stream, verbose=True, user_request_args=self._user_request_args)
        if self._task_status_queue and (self._total_file_count > 0 or self._total_size > 0):
            progress_callbacks.workload_estimator_callback(self._task_status_queue, item_count=self._total_file_count, size=self._total_size)
        if self._total_size > _ONE_TB_IN_BYTES and self._has_cloud_source and (not self._has_local_source) and (self._raw_destination.storage_url.scheme is storage_url.ProviderPrefix.GCS) and properties.VALUES.storage.suggest_transfer.GetBool():
            log.status.Print('For large copies, consider the `gcloud transfer jobs create ...` command. Learn more at\nhttps://cloud.google.com/storage-transfer-service\nRun `gcloud config set storage/suggest_transfer False` to disable this message.')

    def _get_copy_destination(self, raw_destination, source):
        """Returns the final destination StorageUrl instance."""
        completion_is_necessary = _destination_is_container(raw_destination) or (self._multiple_sources and (not _resource_is_stream(raw_destination))) or source.resource.storage_url.versionless_url_string != source.expanded_url.versionless_url_string
        if completion_is_necessary:
            if isinstance(source.expanded_url, storage_url.FileUrl) and source.expanded_url.is_stdio:
                raise errors.Error('Destination object name needed when source is stdin.')
            destination_resource = self._complete_destination(raw_destination, source)
        else:
            destination_resource = raw_destination
        sanitized_destination_resource = path_util.sanitize_file_resource_for_windows(destination_resource)
        return sanitized_destination_resource

    def _complete_destination(self, destination_container, source):
        """Gets a valid copy destination incorporating part of the source's name.

    When given a source file or object and a destination resource that should
    be treated as a container, this function uses the last part of the source's
    name to get an object or file resource representing the copy destination.

    For example: given a source `dir/file` and a destination `gs://bucket/`, the
    destination returned is a resource representing `gs://bucket/file`. Check
    the recursive helper function docstring for details on recursion handling.

    Args:
      destination_container (resource_reference.Resource): The destination
        container.
      source (NameExpansionResult): Represents the source resource and the
        expanded parent url in case of recursion.

    Returns:
      The completed destination, a resource_reference.Resource.
    """
        destination_url = destination_container.storage_url
        source_url = source.resource.storage_url
        if source_url.versionless_url_string != source.expanded_url.versionless_url_string:
            destination_suffix = self._get_destination_suffix_for_recursion(destination_container, source)
        else:
            _, _, url_without_scheme = source_url.versionless_url_string.rpartition(source_url.scheme.value + '://')
            if url_without_scheme.endswith(source_url.delimiter):
                url_without_scheme_and_trailing_delimiter = url_without_scheme[:-len(source_url.delimiter)]
            else:
                url_without_scheme_and_trailing_delimiter = url_without_scheme
            _, _, destination_suffix = url_without_scheme_and_trailing_delimiter.rpartition(source_url.delimiter)
            if url_without_scheme_and_trailing_delimiter != url_without_scheme:
                destination_suffix += source_url.delimiter
        destination_url_prefix = storage_url.storage_url_from_string(destination_url.versionless_url_string.rstrip(destination_url.delimiter))
        new_destination_url = destination_url_prefix.join(destination_suffix)
        return resource_reference.UnknownResource(new_destination_url)

    def _get_destination_suffix_for_recursion(self, destination_container, source):
        """Returns the suffix required to complete the destination URL.

    Let's assume the following:
      User command => cp -r */base_dir gs://dest/existing_prefix
      source.resource.storage_url => a/base_dir/c/d.txt
      source.expanded_url => a/base_dir
      destination_container.storage_url => gs://dest/existing_prefix

    If the destination container exists, the entire directory gets copied:
    Result => gs://dest/existing_prefix/base_dir/c/d.txt

    Args:
      destination_container (resource_reference.Resource): The destination
        container.
      source (NameExpansionResult): Represents the source resource and the
        expanded parent url in case of recursion.

    Returns:
      (str) The suffix to be appended to the destination container.
    """
        source_prefix_to_ignore = storage_url.rstrip_one_delimiter(source.expanded_url.versionless_url_string, source.expanded_url.delimiter)
        expanded_url_is_valid_parent = _is_expanded_url_valid_parent_dir(source.expanded_url)
        if not expanded_url_is_valid_parent and self._has_multiple_top_level_sources:
            raise errors.InvalidUrlError('Presence of multiple top-level sources and invalid expanded URL make file name conflicts possible for URL: {}'.format(source.resource))
        is_top_level_source_object_name_conflict_possible = isinstance(destination_container, resource_reference.UnknownResource) and self._has_multiple_top_level_sources
        destination_exists = not isinstance(destination_container, resource_reference.UnknownResource)
        destination_is_existing_dir = destination_exists and destination_container.is_container()
        treat_destination_as_existing_dir = destination_is_existing_dir or (not destination_exists and destination_container.storage_url.url_string.endswith(destination_container.storage_url.delimiter))
        if is_top_level_source_object_name_conflict_possible or (expanded_url_is_valid_parent and treat_destination_as_existing_dir):
            source_delimiter = source.resource.storage_url.delimiter
            relative_path_characters_end_source_prefix = [source_prefix_to_ignore.endswith(source_delimiter + i) for i in _RELATIVE_PATH_SYMBOLS]
            source_url_scheme_string = source.expanded_url.scheme.value + '://'
            source_prefix_to_ignore_without_scheme = source_prefix_to_ignore[len(source_url_scheme_string):]
            source_is_relative_path_symbol = source_prefix_to_ignore_without_scheme in _RELATIVE_PATH_SYMBOLS
            if not any(relative_path_characters_end_source_prefix) and (not source_is_relative_path_symbol):
                source_prefix_to_ignore, _, _ = source_prefix_to_ignore.rpartition(source.expanded_url.delimiter)
            if not source_prefix_to_ignore:
                source_prefix_to_ignore = source.expanded_url.scheme.value + '://'
        full_source_url = source.resource.storage_url.versionless_url_string
        delimiter = source.resource.storage_url.delimiter
        suffix_for_destination = delimiter + full_source_url.split(source_prefix_to_ignore)[1].lstrip(delimiter)
        source_delimiter = source.resource.storage_url.delimiter
        destination_delimiter = destination_container.storage_url.delimiter
        if source_delimiter != destination_delimiter:
            return suffix_for_destination.replace(source_delimiter, destination_delimiter)
        return suffix_for_destination