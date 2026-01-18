from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import folder_util
from googlecloudsdk.command_lib.storage import plurality_checkable_iterator
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import shim_format_util
import six
class BaseListExecutor(six.with_metaclass(abc.ABCMeta)):
    """Abstract base class for list executors (e.g. for ls and du)."""

    def __init__(self, cloud_urls, buckets_flag=False, display_detail=DisplayDetail.SHORT, exclude_patterns=None, fetch_encrypted_object_hashes=False, halt_on_empty_response=True, include_etag=False, include_managed_folders=False, next_page_token=None, object_state=None, readable_sizes=False, recursion_flag=False, total=False, use_gsutil_style=False, zero_terminator=False):
        """Initializes executor.

    Args:
      cloud_urls ([storage_url.CloudUrl]): List of non-local filesystem URLs.
      buckets_flag (bool): If given a bucket URL, only return matching buckets
        ignoring normal recursion rules.
      display_detail (DisplayDetail): Determines level of metadata printed.
      exclude_patterns (Patterns|None): Don't return resources whose URLs or
        local file paths matched these regex patterns.
      fetch_encrypted_object_hashes (bool): Fall back to GET requests for
        encrypted objects in order to fetch their hash values.
      halt_on_empty_response (bool): Stops querying after empty list response.
        See CloudApi for details.
      include_etag (bool): Print etag string of resource, depending on other
        settings.
      include_managed_folders (bool): Includes managed folders in list results.
      next_page_token (str|None): Used to resume LIST calls.
      object_state (cloud_api.ObjectState): Versions of objects to query.
      readable_sizes (bool): Convert bytes to a more human readable format for
        long lising. For example, print 1024B as 1KiB.
      recursion_flag (bool): Recurse through all containers and format all
        container headers.
      total (bool): Add up the total size of all input sources.
      use_gsutil_style (bool): Outputs closer to the style of the gsutil CLI.
      zero_terminator (bool): Use null byte instead of newline as line
        terminator.
    """
        self._cloud_urls = cloud_urls
        self._buckets_flag = buckets_flag
        self._display_detail = display_detail
        self._exclude_patterns = exclude_patterns
        self._fetch_encrypted_object_hashes = fetch_encrypted_object_hashes
        self._halt_on_empty_response = halt_on_empty_response
        self._include_etag = include_etag
        self._include_managed_folders = include_managed_folders
        self._next_page_token = next_page_token
        self._object_state = object_state
        self._readable_sizes = readable_sizes
        self._recursion_flag = recursion_flag
        self._total = total
        self._use_gsutil_style = use_gsutil_style
        self._zero_terminator = zero_terminator
        self._full_formatter = None
        self._header_wrapper = NullFormatWrapper
        self._container_summary_wrapper = NullFormatWrapper
        self._object_wrapper = NullFormatWrapper

    def _get_container_iterator(self, container_cloud_url, recursion_level):
        """For recursing into and retrieving the contents of a container.

    Args:
      container_cloud_url (storage_url.CloudUrl): Container URL for recursing
        into.
      recursion_level (int): Determines if iterator should keep recursing.

    Returns:
      BaseFormatWrapper generator.
    """
        new_cloud_url = container_cloud_url.join('*')
        fields_scope = _translate_display_detail_to_fields_scope(self._display_detail, is_bucket_listing=False)
        if self._include_managed_folders:
            managed_folder_setting = folder_util.ManagedFolderSetting.LIST_AS_PREFIXES
        else:
            managed_folder_setting = folder_util.ManagedFolderSetting.DO_NOT_LIST
        iterator = wildcard_iterator.CloudWildcardIterator(new_cloud_url, error_on_missing_key=False, exclude_patterns=self._exclude_patterns, fetch_encrypted_object_hashes=self._fetch_encrypted_object_hashes, fields_scope=fields_scope, halt_on_empty_response=self._halt_on_empty_response, managed_folder_setting=managed_folder_setting, next_page_token=self._next_page_token, object_state=self._object_state)
        return self._recursion_helper(iterator, recursion_level)

    def _recursion_helper(self, iterator, recursion_level, print_top_level_container=True):
        """For retrieving resources from URLs that potentially contain wildcards.

    Args:
      iterator (Iterable[resource_reference.Resource]): For recursing through.
      recursion_level (int): Integer controlling how deep the listing recursion
        goes. "1" is the default, mimicking the actual OS ls, which lists the
        contents of the first level of matching subdirectories. Call with
        "float('inf')" for listing everything available.
      print_top_level_container (bool): Used by `du` to skip printing the top
        level bucket

    Yields:
      BaseFormatWrapper generator.
    """
        for resource in iterator:
            if resource_reference.is_container_or_has_container_url(resource) and recursion_level > 0:
                if self._header_wrapper != NullFormatWrapper:
                    yield self._header_wrapper(resource, display_detail=self._display_detail, include_etag=self._include_etag, object_state=self._object_state, readable_sizes=self._readable_sizes, full_formatter=self._full_formatter, use_gsutil_style=self._use_gsutil_style)
                container_size = 0
                nested_iterator = self._get_container_iterator(resource.storage_url, recursion_level - 1)
                for nested_resource in nested_iterator:
                    if self._container_summary_wrapper != NullFormatWrapper and print_top_level_container:
                        container_size += getattr(nested_resource.resource, 'size', 0)
                    yield nested_resource
                if self._container_summary_wrapper != NullFormatWrapper and print_top_level_container:
                    yield self._container_summary_wrapper(resource=resource, container_size=container_size, object_state=self._object_state, readable_sizes=self._readable_sizes)
            else:
                yield self._object_wrapper(resource, display_detail=self._display_detail, full_formatter=self._full_formatter, include_etag=self._include_etag, object_state=self._object_state, readable_sizes=self._readable_sizes, use_gsutil_style=self._use_gsutil_style)

    def _print_summary_for_top_level_url(self, resource_url, only_display_buckets, object_count, total_bytes):
        del self, resource_url, only_display_buckets, object_count, total_bytes

    def _print_row_list(self, resource_wrappers, resource_url, only_display_buckets):
        """Prints ResourceWrapper objects in list with custom row formatting."""
        object_count = total_bytes = 0
        terminator = '\x00' if self._zero_terminator else '\n'
        for i, resource_wrapper in enumerate(resource_wrappers):
            resource_wrapper_string = six.text_type(resource_wrapper)
            if isinstance(resource_wrapper.resource, resource_reference.ObjectResource):
                object_count += 1
                total_bytes += resource_wrapper.resource.size or 0
            if not resource_wrapper_string:
                continue
            if i == 0 and resource_wrapper and (resource_wrapper_string[0] == '\n'):
                print(resource_wrapper_string[1:], end=terminator)
            else:
                print(resource_wrapper_string, end=terminator)
        self._print_summary_for_top_level_url(resource_url=resource_url, only_display_buckets=only_display_buckets, object_count=object_count, total_bytes=total_bytes)
        return total_bytes

    def _should_only_display_buckets(self, raw_cloud_url):
        return raw_cloud_url.is_provider() or (self._buckets_flag and raw_cloud_url.is_bucket())

    def _list_url(self, raw_cloud_url):
        """Recursively create wildcard iterators to print all relevant items."""
        fields_scope = _translate_display_detail_to_fields_scope(self._display_detail, is_bucket_listing=raw_cloud_url.is_provider())
        if self._include_managed_folders:
            managed_folder_setting = folder_util.ManagedFolderSetting.LIST_AS_PREFIXES
        else:
            managed_folder_setting = folder_util.ManagedFolderSetting.DO_NOT_LIST
        resources = plurality_checkable_iterator.PluralityCheckableIterator(wildcard_iterator.CloudWildcardIterator(raw_cloud_url, error_on_missing_key=False, exclude_patterns=self._exclude_patterns, fetch_encrypted_object_hashes=self._fetch_encrypted_object_hashes, fields_scope=fields_scope, get_bucket_metadata=self._buckets_flag, halt_on_empty_response=self._halt_on_empty_response, managed_folder_setting=managed_folder_setting, next_page_token=self._next_page_token, object_state=self._object_state))
        if resources.is_empty():
            raise errors.InvalidUrlError('One or more URLs matched no objects.')
        only_display_buckets = self._should_only_display_buckets(raw_cloud_url)
        if only_display_buckets:
            resources_wrappers = self._recursion_helper(resources, recursion_level=0)
        elif self._recursion_flag and '**' not in raw_cloud_url.url_string:
            print_top_level_container = True
            if raw_cloud_url.is_bucket():
                print_top_level_container = False
            resources_wrappers = self._recursion_helper(resources, float('inf'), print_top_level_container)
        elif not resources.is_plural() and resource_reference.is_container_or_has_container_url(resources.peek()):
            resources_wrappers = self._get_container_iterator(resources.peek().storage_url, recursion_level=0)
        else:
            resources_wrappers = self._recursion_helper(resources, recursion_level=1)
        size_in_bytes = 0
        if self._display_detail == DisplayDetail.JSON:
            self._print_json_list(resources_wrappers)
        else:
            size_in_bytes = self._print_row_list(resources_wrappers, raw_cloud_url, only_display_buckets)
        return size_in_bytes

    def _print_total(self, all_sources_total_bytes):
        del all_sources_total_bytes

    def list_urls(self):
        all_sources_total_bytes = 0
        for url in self._cloud_urls:
            if self._total:
                all_sources_total_bytes += self._list_url(url)
            else:
                self._list_url(url)
        if self._total:
            self._print_total(all_sources_total_bytes)