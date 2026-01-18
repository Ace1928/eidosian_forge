from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import itertools
import json
import os
import re
import subprocess
import textwrap
import six
from six.moves import zip
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite.messages import DecodeError
from boto import config
from gslib.cloud_api import ArgumentException
from gslib.cloud_api import PreconditionException
from gslib.cloud_api import ServiceException
from gslib.command import Command
from gslib.command import GetFailureCount
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.exception import IamChOnResourceWithConditionsException
from gslib.help_provider import CreateHelpText
from gslib.metrics import LogCommandParams
from gslib.name_expansion import NameExpansionIterator
from gslib.name_expansion import SeekAheadNameExpansionIterator
from gslib.plurality_checkable_iterator import PluralityCheckableIterator
from gslib.storage_url import GetSchemeFromUrlString
from gslib.storage_url import IsKnownUrlScheme
from gslib.storage_url import StorageUrlFromString
from gslib.storage_url import UrlsAreMixOfBucketsAndObjects
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import shim_util
from gslib.utils.cloud_api_helper import GetCloudApiInstance
from gslib.utils.constants import IAM_POLICY_VERSION
from gslib.utils.constants import NO_MAX
from gslib.utils import iam_helper
from gslib.utils.iam_helper import BindingStringToTuple
from gslib.utils.iam_helper import BindingsTuple
from gslib.utils.iam_helper import DeserializeBindingsTuple
from gslib.utils.iam_helper import IsEqualBindings
from gslib.utils.iam_helper import PatchBindings
from gslib.utils.iam_helper import SerializeBindingsTuple
from gslib.utils.retry_util import Retry
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.shim_util import GcloudStorageFlag
def _PatchIam(self):
    raw_bindings_tuples, url_patterns = self._GetSettingsAndDiffs()
    patch_bindings_tuples = []
    for is_grant, bindings in raw_bindings_tuples:
        bindings_messages = []
        for binding in bindings:
            bindings_message = apitools_messages.Policy.BindingsValueListEntry(members=binding['members'], role=binding['role'])
            bindings_messages.append(bindings_message)
        patch_bindings_tuples.append(BindingsTuple(is_grant=is_grant, bindings=bindings_messages))
    self.everything_set_okay = True
    self.tried_ch_on_resource_with_conditions = False
    threaded_wildcards = []
    for surl in url_patterns:
        try:
            if surl.IsBucket():
                if self.recursion_requested:
                    surl.object = '*'
                    threaded_wildcards.append(surl.url_string)
                else:
                    self.PatchIamHelper(surl, patch_bindings_tuples)
            else:
                threaded_wildcards.append(surl.url_string)
        except AttributeError:
            self._RaiseIfInvalidUrl(surl)
    if threaded_wildcards:
        name_expansion_iterator = NameExpansionIterator(self.command_name, self.debug, self.logger, self.gsutil_api, threaded_wildcards, self.recursion_requested, all_versions=self.all_versions, continue_on_error=self.continue_on_error or self.parallel_operations, bucket_listing_fields=['name'])
        seek_ahead_iterator = SeekAheadNameExpansionIterator(self.command_name, self.debug, self.GetSeekAheadGsutilApi(), threaded_wildcards, self.recursion_requested, all_versions=self.all_versions)
        serialized_bindings_tuples_it = itertools.repeat([SerializeBindingsTuple(t) for t in patch_bindings_tuples])
        self.Apply(_PatchIamWrapper, zip(serialized_bindings_tuples_it, name_expansion_iterator), _PatchIamExceptionHandler, fail_on_error=not self.continue_on_error, seek_ahead_iterator=seek_ahead_iterator)
        self.everything_set_okay &= not GetFailureCount() > 0
    if not self.everything_set_okay:
        msg = 'Some IAM policies could not be patched.'
        if self.tried_ch_on_resource_with_conditions:
            msg += '\n'
            msg += '\n'.join(textwrap.wrap('Some resources had conditions present in their IAM policy bindings, which is not supported by "iam ch". %s' % IAM_CH_CONDITIONS_WORKAROUND_MSG))
        raise CommandException(msg)