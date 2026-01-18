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
def _SetIam(self):
    """Set IAM policy for given wildcards on the command line."""
    self.continue_on_error = False
    self.recursion_requested = False
    self.all_versions = False
    force_etag = False
    etag = ''
    if self.sub_opts:
        for o, arg in self.sub_opts:
            if o in ['-r', '-R']:
                self.recursion_requested = True
            elif o == '-f':
                self.continue_on_error = True
            elif o == '-a':
                self.all_versions = True
            elif o == '-e':
                etag = str(arg)
                force_etag = True
            else:
                self.RaiseInvalidArgumentException()
    file_url = self.args[0]
    patterns = self.args[1:]
    try:
        with open(file_url, 'r') as fp:
            policy = json.loads(fp.read())
    except IOError:
        raise ArgumentException('Specified IAM policy file "%s" does not exist.' % file_url)
    except ValueError as e:
        self.logger.debug('Invalid IAM policy file, ValueError:\n%s', e)
        raise ArgumentException('Invalid IAM policy file "%s".' % file_url)
    bindings = policy.get('bindings', [])
    if not force_etag:
        etag = policy.get('etag', '')
    policy_json = json.dumps({'bindings': bindings, 'etag': etag, 'version': IAM_POLICY_VERSION})
    try:
        policy = protojson.decode_message(apitools_messages.Policy, policy_json)
    except DecodeError:
        raise ArgumentException('Invalid IAM policy file "%s" or etag "%s".' % (file_url, etag))
    self.everything_set_okay = True
    threaded_wildcards = []
    surls = list(map(StorageUrlFromString, patterns))
    _RaiseErrorIfUrlsAreMixOfBucketsAndObjects(surls, self.recursion_requested)
    for surl in surls:
        print(surl.url_string)
        if surl.IsBucket():
            if self.recursion_requested:
                surl.object_name = '*'
                threaded_wildcards.append(surl.url_string)
            else:
                self.SetIamHelper(surl, policy)
        else:
            threaded_wildcards.append(surl.url_string)
    if threaded_wildcards:
        name_expansion_iterator = NameExpansionIterator(self.command_name, self.debug, self.logger, self.gsutil_api, threaded_wildcards, self.recursion_requested, all_versions=self.all_versions, continue_on_error=self.continue_on_error or self.parallel_operations, bucket_listing_fields=['name'])
        seek_ahead_iterator = SeekAheadNameExpansionIterator(self.command_name, self.debug, self.GetSeekAheadGsutilApi(), threaded_wildcards, self.recursion_requested, all_versions=self.all_versions)
        policy_it = itertools.repeat(protojson.encode_message(policy))
        self.Apply(_SetIamWrapper, zip(policy_it, name_expansion_iterator), _SetIamExceptionHandler, fail_on_error=not self.continue_on_error, seek_ahead_iterator=seek_ahead_iterator)
        self.everything_set_okay &= not GetFailureCount() > 0
    if not self.everything_set_okay:
        raise CommandException('Some IAM policies could not be set.')