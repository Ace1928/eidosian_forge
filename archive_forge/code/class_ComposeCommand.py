from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from boto import config
from gslib.bucket_listing_ref import BucketListingObject
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.storage_url import ContainsWildcard
from gslib.storage_url import StorageUrlFromString
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils.encryption_helper import GetEncryptionKeyWrapper
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.translation_helper import PreconditionsFromHeaders
class ComposeCommand(Command):
    """Implementation of gsutil compose command."""
    command_spec = Command.CreateCommandSpec('compose', command_name_aliases=['concat'], usage_synopsis=_SYNOPSIS, min_args=1, max_args=MAX_COMPOSE_ARITY + 1, supported_sub_args='', file_url_ok=False, provider_url_ok=False, urls_start_arg=1, gs_api_support=[ApiSelector.XML, ApiSelector.JSON], gs_default_api=ApiSelector.JSON, argparse_arguments=[CommandArgument.MakeZeroOrMoreCloudURLsArgument()])
    help_spec = Command.HelpSpec(help_name='compose', help_name_aliases=['concat'], help_type='command_help', help_one_line_summary='Concatenate a sequence of objects into a new composite object.', help_text=_DETAILED_HELP_TEXT, subcommand_help_text={})
    gcloud_storage_map = GcloudStorageMap(gcloud_command=['storage', 'objects', 'compose'], flag_map={})

    def CheckProvider(self, url):
        if url.scheme != 'gs':
            raise CommandException('"compose" called on URL with unsupported provider (%s).' % str(url))

    def RunCommand(self):
        """Command entry point for the compose command."""
        target_url_str = self.args[-1]
        self.args = self.args[:-1]
        target_url = StorageUrlFromString(target_url_str)
        self.CheckProvider(target_url)
        if target_url.HasGeneration():
            raise CommandException('A version-specific URL (%s) cannot be the destination for gsutil compose - abort.' % target_url)
        dst_obj_metadata = apitools_messages.Object(name=target_url.object_name, bucket=target_url.bucket_name)
        components = []
        first_src_url = None
        for src_url_str in self.args:
            if ContainsWildcard(src_url_str):
                src_url_iter = self.WildcardIterator(src_url_str).IterObjects()
            else:
                src_url_iter = [BucketListingObject(StorageUrlFromString(src_url_str))]
            for blr in src_url_iter:
                src_url = blr.storage_url
                self.CheckProvider(src_url)
                if src_url.bucket_name != target_url.bucket_name:
                    raise CommandException('GCS does not support inter-bucket composing.')
                if not first_src_url:
                    first_src_url = src_url
                src_obj_metadata = apitools_messages.ComposeRequest.SourceObjectsValueListEntry(name=src_url.object_name)
                if src_url.HasGeneration():
                    src_obj_metadata.generation = int(src_url.generation)
                components.append(src_obj_metadata)
                if len(components) > MAX_COMPOSE_ARITY:
                    raise CommandException('"compose" called with too many component objects. Limit is %d.' % MAX_COMPOSE_ARITY)
        if not components:
            raise CommandException('"compose" requires at least 1 component object.')
        first_src_obj_metadata = self.gsutil_api.GetObjectMetadata(first_src_url.bucket_name, first_src_url.object_name, provider=first_src_url.scheme, fields=['contentEncoding', 'contentType'])
        dst_obj_metadata.contentType = first_src_obj_metadata.contentType
        dst_obj_metadata.contentEncoding = first_src_obj_metadata.contentEncoding
        preconditions = PreconditionsFromHeaders(self.headers or {})
        self.logger.info('Composing %s from %d component object(s).', target_url, len(components))
        self.gsutil_api.ComposeObject(components, dst_obj_metadata, preconditions=preconditions, provider=target_url.scheme, encryption_tuple=GetEncryptionKeyWrapper(config))