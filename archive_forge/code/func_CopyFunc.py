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