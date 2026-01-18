from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import errno
import itertools
import logging
import os
import time
import traceback
from apitools.base.py import encoding
from gslib import gcs_json_api
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.metrics import LogPerformanceSummaryParams
from gslib.name_expansion import CopyObjectsIterator
from gslib.name_expansion import DestinationInfo
from gslib.name_expansion import NameExpansionIterator
from gslib.name_expansion import NameExpansionIteratorDestinationTuple
from gslib.name_expansion import SeekAheadNameExpansionIterator
from gslib.storage_url import ContainsWildcard
from gslib.storage_url import IsCloudSubdirPlaceholder
from gslib.storage_url import StorageUrlFromString
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import cat_helper
from gslib.utils import copy_helper
from gslib.utils import parallelism_framework_util
from gslib.utils.cloud_api_helper import GetCloudApiInstance
from gslib.utils.constants import DEBUGLEVEL_DUMP_REQUESTS
from gslib.utils.constants import NO_MAX
from gslib.utils.copy_helper import CreateCopyHelperOpts
from gslib.utils.copy_helper import GetSourceFieldsNeededForCopy
from gslib.utils.copy_helper import GZIP_ALL_FILES
from gslib.utils.copy_helper import ItemExistsError
from gslib.utils.copy_helper import Manifest
from gslib.utils.copy_helper import SkipUnsupportedObjectError
from gslib.utils.posix_util import ConvertModeToBase8
from gslib.utils.posix_util import DeserializeFileAttributesFromObjectMetadata
from gslib.utils.posix_util import InitializePreservePosixData
from gslib.utils.posix_util import POSIXAttributes
from gslib.utils.posix_util import SerializeFileAttributesToObjectMetadata
from gslib.utils.posix_util import ValidateFilePermissionAccess
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.system_util import GetStreamFromFileUrl
from gslib.utils.system_util import StdinIterator
from gslib.utils.system_util import StdinIteratorCls
from gslib.utils.text_util import NormalizeStorageClass
from gslib.utils.text_util import RemoveCRLFFromString
from gslib.utils.unit_util import CalculateThroughput
from gslib.utils.unit_util import MakeHumanReadable
class CpCommand(Command):
    """Implementation of gsutil cp command.

  Note that CpCommand is run for both gsutil cp and gsutil mv. The latter
  happens by MvCommand calling CpCommand and passing the hidden (undocumented)
  -M option. This allows the copy and remove needed for each mv to run
  together (rather than first running all the cp's and then all the rm's, as
  we originally had implemented), which in turn avoids the following problem
  with removing the wrong objects: starting with a bucket containing only
  the object gs://bucket/obj, say the user does:
    gsutil mv gs://bucket/* gs://bucket/d.txt
  If we ran all the cp's and then all the rm's and we didn't expand the wildcard
  first, the cp command would first copy gs://bucket/obj to gs://bucket/d.txt,
  and the rm command would then remove that object. In the implementation
  prior to gsutil release 3.12 we avoided this by building a list of objects
  to process and then running the copies and then the removes; but building
  the list up front limits scalability (compared with the current approach
  of processing the bucket listing iterator on the fly).
  """
    command_spec = Command.CreateCommandSpec('cp', command_name_aliases=['copy'], usage_synopsis=_SYNOPSIS, min_args=1, max_args=NO_MAX, supported_sub_args=CP_SUB_ARGS, file_url_ok=True, provider_url_ok=False, urls_start_arg=0, gs_api_support=[ApiSelector.XML, ApiSelector.JSON], gs_default_api=ApiSelector.JSON, supported_private_args=['stet', 'testcallbackfile='], argparse_arguments=[CommandArgument.MakeZeroOrMoreCloudOrFileURLsArgument()])
    help_spec = Command.HelpSpec(help_name='cp', help_name_aliases=['copy'], help_type='command_help', help_one_line_summary='Copy files and objects', help_text=_DETAILED_HELP_TEXT, subcommand_help_text={})

    def get_gcloud_storage_args(self):
        self.logger.warn('Unlike pure gsutil, this shim won\'t run composite uploads and sliced downloads in parallel by default. Use the -m flag to enable parallelism (i.e. "gsutil -m cp ...").')
        ShimTranslatePredefinedAclSubOptForCopy(self.sub_opts)
        gcloud_storage_map = GcloudStorageMap(gcloud_command=['storage', 'cp'], flag_map=CP_SHIM_FLAG_MAP)
        return super().get_gcloud_storage_args(gcloud_storage_map)

    def CopyFunc(self, copy_object_info, thread_state=None, preserve_posix=False):
        """Worker function for performing the actual copy (and rm, for mv)."""
        gsutil_api = GetCloudApiInstance(self, thread_state=thread_state)
        copy_helper_opts = copy_helper.GetCopyHelperOpts()
        if copy_helper_opts.perform_mv:
            cmd_name = 'mv'
        else:
            cmd_name = self.command_name
        src_url = copy_object_info.source_storage_url
        exp_src_url = copy_object_info.expanded_storage_url
        src_url_names_container = copy_object_info.names_container
        have_multiple_srcs = copy_object_info.is_multi_source_request
        if src_url.IsCloudUrl() and src_url.IsProvider():
            raise CommandException('The %s command does not allow provider-only source URLs (%s)' % (cmd_name, src_url))
        if preserve_posix and src_url.IsFileUrl() and src_url.IsStream():
            raise CommandException('Cannot preserve POSIX attributes with a stream.')
        if self.parallel_operations and src_url.IsFileUrl() and src_url.IsStream():
            raise CommandException('Cannot upload from a stream when using gsutil -m option.')
        if have_multiple_srcs:
            copy_helper.InsistDstUrlNamesContainer(copy_object_info.exp_dst_url, copy_object_info.have_existing_dst_container, cmd_name)
        if IsCloudSubdirPlaceholder(exp_src_url):
            return
        if copy_helper_opts.use_manifest and self.manifest.WasSuccessful(exp_src_url.url_string):
            return
        if copy_helper_opts.perform_mv and copy_object_info.names_container:
            self.recursion_requested = True
        if copy_object_info.exp_dst_url.IsFileUrl() and (not os.path.exists(copy_object_info.exp_dst_url.object_name)) and have_multiple_srcs:
            try:
                os.makedirs(copy_object_info.exp_dst_url.object_name)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        dst_url = copy_helper.ConstructDstUrl(src_url, exp_src_url, src_url_names_container, have_multiple_srcs, copy_object_info.is_multi_top_level_source_request, copy_object_info.exp_dst_url, copy_object_info.have_existing_dst_container, self.recursion_requested, preserve_posix=preserve_posix)
        dst_url = copy_helper.FixWindowsNaming(src_url, dst_url)
        copy_helper.CheckForDirFileConflict(exp_src_url, dst_url)
        if copy_helper.SrcDstSame(exp_src_url, dst_url):
            raise CommandException('%s: "%s" and "%s" are the same file - abort.' % (cmd_name, exp_src_url, dst_url))
        if dst_url.IsCloudUrl() and dst_url.HasGeneration():
            raise CommandException('%s: a version-specific URL\n(%s)\ncannot be the destination for gsutil cp - abort.' % (cmd_name, dst_url))
        if not dst_url.IsCloudUrl() and copy_helper_opts.dest_storage_class:
            raise CommandException('Cannot specify storage class for a non-cloud destination: %s' % dst_url)
        src_obj_metadata = None
        if copy_object_info.expanded_result:
            src_obj_metadata = encoding.JsonToMessage(apitools_messages.Object, copy_object_info.expanded_result)
        if src_url.IsFileUrl() and preserve_posix:
            if not src_obj_metadata:
                src_obj_metadata = apitools_messages.Object()
            mode, _, _, _, uid, gid, _, atime, mtime, _ = os.stat(exp_src_url.object_name)
            mode = ConvertModeToBase8(mode)
            posix_attrs = POSIXAttributes(atime=atime, mtime=mtime, uid=uid, gid=gid, mode=mode)
            custom_metadata = apitools_messages.Object.MetadataValue(additionalProperties=[])
            SerializeFileAttributesToObjectMetadata(posix_attrs, custom_metadata, preserve_posix=preserve_posix)
            src_obj_metadata.metadata = custom_metadata
        if src_obj_metadata and dst_url.IsFileUrl():
            posix_attrs = DeserializeFileAttributesFromObjectMetadata(src_obj_metadata, src_url.url_string)
            mode = posix_attrs.mode.permissions
            valid, err = ValidateFilePermissionAccess(src_url.url_string, uid=posix_attrs.uid, gid=posix_attrs.gid, mode=mode)
            if preserve_posix and (not valid):
                logging.getLogger().critical(err)
                raise CommandException('This sync will orphan file(s), please fix their permissions before trying again.')
        bytes_transferred = 0
        try:
            if copy_helper_opts.use_manifest:
                self.manifest.Initialize(exp_src_url.url_string, dst_url.url_string)
            if self.recursion_requested and copy_object_info.exp_dst_url.object_name and dst_url.IsFileUrl():
                container = os.path.abspath(copy_object_info.exp_dst_url.object_name)
                if not os.path.commonpath([container, os.path.abspath(dst_url.object_name)]).startswith(container):
                    self.logger.warn('Skipping copy of source URL %s because it would be copied outside the expected destination directory: %s.' % (exp_src_url, container))
                    if copy_helper_opts.use_manifest:
                        self.manifest.SetResult(exp_src_url.url_string, 0, 'skip', 'Would have copied outside the destination directory.')
                    return
            _, bytes_transferred, result_url, md5 = copy_helper.PerformCopy(self.logger, exp_src_url, dst_url, gsutil_api, self, _CopyExceptionHandler, src_obj_metadata=src_obj_metadata, allow_splitting=True, headers=self.headers, manifest=self.manifest, gzip_encoded=self.gzip_encoded, gzip_exts=self.gzip_exts, preserve_posix=preserve_posix, use_stet=self.use_stet)
            if copy_helper_opts.use_manifest:
                if md5:
                    self.manifest.Set(exp_src_url.url_string, 'md5', md5)
                self.manifest.SetResult(exp_src_url.url_string, bytes_transferred, 'OK')
            if copy_helper_opts.print_ver:
                self.logger.info('Created: %s', result_url)
        except ItemExistsError:
            message = 'Skipping existing item: %s' % dst_url
            self.logger.info(message)
            if copy_helper_opts.use_manifest:
                self.manifest.SetResult(exp_src_url.url_string, 0, 'skip', message)
        except SkipUnsupportedObjectError as e:
            message = 'Skipping item %s with unsupported object type %s' % (exp_src_url.url_string, e.unsupported_type)
            self.logger.info(message)
            if copy_helper_opts.use_manifest:
                self.manifest.SetResult(exp_src_url.url_string, 0, 'skip', message)
        except copy_helper.FileConcurrencySkipError as e:
            self.logger.warn('Skipping copy of source URL %s because destination URL %s is already being copied by another gsutil process or thread (did you specify the same source URL twice?) ' % (src_url, dst_url))
        except Exception as e:
            if copy_helper_opts.no_clobber and copy_helper.IsNoClobberServerException(e):
                message = 'Rejected (noclobber): %s' % dst_url
                self.logger.info(message)
                if copy_helper_opts.use_manifest:
                    self.manifest.SetResult(exp_src_url.url_string, 0, 'skip', message)
            elif self.continue_on_error:
                message = 'Error copying %s: %s' % (src_url, str(e))
                self.op_failure_count += 1
                self.logger.error(message)
                if copy_helper_opts.use_manifest:
                    self.manifest.SetResult(exp_src_url.url_string, 0, 'error', RemoveCRLFFromString(message))
            else:
                if copy_helper_opts.use_manifest:
                    self.manifest.SetResult(exp_src_url.url_string, 0, 'error', str(e))
                raise
        else:
            if copy_helper_opts.perform_mv:
                self.logger.info('Removing %s...', exp_src_url)
                if exp_src_url.IsCloudUrl():
                    gsutil_api.DeleteObject(exp_src_url.bucket_name, exp_src_url.object_name, generation=exp_src_url.generation, provider=exp_src_url.scheme)
                else:
                    os.unlink(exp_src_url.object_name)
        with self.stats_lock:
            self.total_bytes_transferred += bytes_transferred

    def _ConstructNameExpansionIteratorDstTupleIterator(self, src_url_strs_iter, dst_url_strs):
        copy_helper_opts = copy_helper.GetCopyHelperOpts()
        for src_url_str, dst_url_str in zip(src_url_strs_iter, dst_url_strs):
            exp_dst_url, have_existing_dst_container = copy_helper.ExpandUrlToSingleBlr(dst_url_str, self.gsutil_api, self.project_id, logger=self.logger)
            name_expansion_iterator_dst_tuple = NameExpansionIteratorDestinationTuple(NameExpansionIterator(self.command_name, self.debug, self.logger, self.gsutil_api, src_url_str, self.recursion_requested or copy_helper_opts.perform_mv, project_id=self.project_id, all_versions=self.all_versions, ignore_symlinks=self.exclude_symlinks, continue_on_error=self.continue_on_error or self.parallel_operations, bucket_listing_fields=GetSourceFieldsNeededForCopy(exp_dst_url.IsCloudUrl(), copy_helper_opts.skip_unsupported_objects, copy_helper_opts.preserve_acl, preserve_posix=self.preserve_posix_attrs, delete_source=copy_helper_opts.perform_mv, file_size_will_change=self.use_stet)), DestinationInfo(exp_dst_url, have_existing_dst_container))
            self.has_file_dst = self.has_file_dst or exp_dst_url.IsFileUrl()
            self.has_cloud_dst = self.has_cloud_dst or exp_dst_url.IsCloudUrl()
            self.provider_types.add(exp_dst_url.scheme)
            self.combined_src_urls = itertools.chain(self.combined_src_urls, src_url_str)
            yield name_expansion_iterator_dst_tuple

    def RunCommand(self):
        copy_helper_opts = self._ParseOpts()
        self.total_bytes_transferred = 0
        dst_url = StorageUrlFromString(self.args[-1])
        if dst_url.IsFileUrl() and (dst_url.object_name == '-' or dst_url.IsFifo()):
            if self.preserve_posix_attrs:
                raise CommandException('Cannot preserve POSIX attributes with a stream or a named pipe.')
            cat_out_fd = GetStreamFromFileUrl(dst_url, mode='wb') if dst_url.IsFifo() else None
            return cat_helper.CatHelper(self).CatUrlStrings(self.args[:-1], cat_out_fd=cat_out_fd)
        if copy_helper_opts.read_args_from_stdin:
            if len(self.args) != 1:
                raise CommandException('Source URLs cannot be specified with -I option')
            src_url_strs = [StdinIteratorCls()]
        else:
            if len(self.args) < 2:
                raise CommandException('Wrong number of arguments for "cp" command.')
            src_url_strs = [self.args[:-1]]
        dst_url_strs = [self.args[-1]]
        self.combined_src_urls = []
        self.has_file_dst = False
        self.has_cloud_dst = False
        self.provider_types = set()
        name_expansion_iterator = CopyObjectsIterator(self._ConstructNameExpansionIteratorDstTupleIterator(src_url_strs, dst_url_strs), copy_helper_opts.daisy_chain)
        process_count, thread_count = self._GetProcessAndThreadCount(process_count=None, thread_count=None, parallel_operations_override=None, print_macos_warning=False)
        copy_helper.TriggerReauthForDestinationProviderIfNecessary(dst_url, self.gsutil_api, process_count * thread_count)
        seek_ahead_iterator = None
        if not copy_helper_opts.read_args_from_stdin:
            seek_ahead_iterator = SeekAheadNameExpansionIterator(self.command_name, self.debug, self.GetSeekAheadGsutilApi(), self.combined_src_urls, self.recursion_requested or copy_helper_opts.perform_mv, all_versions=self.all_versions, project_id=self.project_id, ignore_symlinks=self.exclude_symlinks, file_size_will_change=self.use_stet)
        self.stats_lock = parallelism_framework_util.CreateLock()
        self.op_failure_count = 0
        start_time = time.time()
        shared_attrs = ('op_failure_count', 'total_bytes_transferred')
        self.Apply(_CopyFuncWrapper, name_expansion_iterator, _CopyExceptionHandler, shared_attrs, fail_on_error=not self.continue_on_error, seek_ahead_iterator=seek_ahead_iterator)
        self.logger.debug('total_bytes_transferred: %d', self.total_bytes_transferred)
        end_time = time.time()
        self.total_elapsed_time = end_time - start_time
        self.total_bytes_per_second = CalculateThroughput(self.total_bytes_transferred, self.total_elapsed_time)
        LogPerformanceSummaryParams(has_file_dst=self.has_file_dst, has_cloud_dst=self.has_cloud_dst, avg_throughput=self.total_bytes_per_second, total_bytes_transferred=self.total_bytes_transferred, total_elapsed_time=self.total_elapsed_time, uses_fan=self.parallel_operations, is_daisy_chain=copy_helper_opts.daisy_chain, provider_types=list(self.provider_types))
        if self.debug >= DEBUGLEVEL_DUMP_REQUESTS:
            if self.total_bytes_transferred != 0:
                self.logger.info('Total bytes copied=%d, total elapsed time=%5.3f secs (%sps)', self.total_bytes_transferred, self.total_elapsed_time, MakeHumanReadable(self.total_bytes_per_second))
        if self.op_failure_count:
            plural_str = 's' if self.op_failure_count > 1 else ''
            raise CommandException('{count} file{pl}/object{pl} could not be transferred.'.format(count=self.op_failure_count, pl=plural_str))
        return 0

    def _ParseOpts(self):
        perform_mv = False
        self.exclude_symlinks = False
        no_clobber = False
        self.continue_on_error = False
        daisy_chain = False
        read_args_from_stdin = False
        print_ver = False
        use_manifest = False
        preserve_acl = False
        self.preserve_posix_attrs = False
        canned_acl = None
        self.canned = None
        self.all_versions = False
        self.skip_unsupported_objects = False
        gzip_encoded = False
        gzip_local = False
        gzip_arg_exts = None
        gzip_arg_all = None
        test_callback_file = None
        dest_storage_class = None
        self.use_stet = False
        self.manifest = None
        if self.sub_opts:
            for o, a in self.sub_opts:
                if o == '-a':
                    canned_acl = a
                    self.canned = True
                if o == '-A':
                    self.all_versions = True
                if o == '-c':
                    self.continue_on_error = True
                elif o == '-D':
                    daisy_chain = True
                elif o == '-e':
                    self.exclude_symlinks = True
                elif o == '--testcallbackfile':
                    test_callback_file = a
                elif o == '-I':
                    read_args_from_stdin = True
                elif o == '-j':
                    gzip_encoded = True
                    gzip_arg_exts = [x.strip() for x in a.split(',')]
                elif o == '-J':
                    gzip_encoded = True
                    gzip_arg_all = GZIP_ALL_FILES
                elif o == '-L':
                    use_manifest = True
                    self.manifest = Manifest(a)
                elif o == '-M':
                    perform_mv = True
                elif o == '-n':
                    no_clobber = True
                elif o == '-p':
                    preserve_acl = True
                elif o == '-P':
                    self.preserve_posix_attrs = True
                    InitializePreservePosixData()
                elif o == '-r' or o == '-R':
                    self.recursion_requested = True
                elif o == '-s':
                    dest_storage_class = NormalizeStorageClass(a)
                elif o == '-U':
                    self.skip_unsupported_objects = True
                elif o == '-v':
                    print_ver = True
                elif o == '-z':
                    gzip_local = True
                    gzip_arg_exts = [x.strip() for x in a.split(',')]
                elif o == '-Z':
                    gzip_local = True
                    gzip_arg_all = GZIP_ALL_FILES
                elif o == '--stet':
                    self.use_stet = True
        if preserve_acl and canned_acl:
            raise CommandException('Specifying both the -p and -a options together is invalid.')
        if self.all_versions and self.parallel_operations:
            raise CommandException('The gsutil -m option is not supported with the cp -A flag, to ensure that object version ordering is preserved. Please re-run the command without the -m option.')
        if gzip_encoded and gzip_local:
            raise CommandException('Specifying both the -j/-J and -z/-Z options together is invalid.')
        if gzip_arg_exts and gzip_arg_all:
            if gzip_encoded:
                raise CommandException('Specifying both the -j and -J options together is invalid.')
            else:
                raise CommandException('Specifying both the -z and -Z options together is invalid.')
        self.gzip_exts = gzip_arg_exts or gzip_arg_all
        self.gzip_encoded = gzip_encoded
        return CreateCopyHelperOpts(perform_mv=perform_mv, no_clobber=no_clobber, daisy_chain=daisy_chain, read_args_from_stdin=read_args_from_stdin, print_ver=print_ver, use_manifest=use_manifest, preserve_acl=preserve_acl, canned_acl=canned_acl, skip_unsupported_objects=self.skip_unsupported_objects, test_callback_file=test_callback_file, dest_storage_class=dest_storage_class)