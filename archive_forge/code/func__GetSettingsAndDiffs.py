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
def _GetSettingsAndDiffs(self):
    self.continue_on_error = False
    self.recursion_requested = False
    patch_bindings_tuples = []
    if self.sub_opts:
        for o, a in self.sub_opts:
            if o in ['-r', '-R']:
                self.recursion_requested = True
            elif o == '-f':
                self.continue_on_error = True
            elif o == '-d':
                patch_bindings_tuples.append(BindingStringToTuple(False, a))
    url_pattern_strings = []
    it = iter(self.args)
    for token in it:
        if STORAGE_URI_REGEX.match(token) and IsKnownUrlScheme(GetSchemeFromUrlString(token)):
            url_pattern_strings.append(token)
            break
        if token == '-d':
            try:
                patch_bindings_tuples.append(BindingStringToTuple(False, next(it)))
            except StopIteration:
                raise CommandException('A -d flag is missing an argument specifying bindings to remove.')
        else:
            patch_bindings_tuples.append(BindingStringToTuple(True, token))
    if not patch_bindings_tuples:
        raise CommandException('Must specify at least one binding.')
    for token in it:
        url_pattern_strings.append(token)
    url_objects = list(map(StorageUrlFromString, url_pattern_strings))
    _RaiseErrorIfUrlsAreMixOfBucketsAndObjects(url_objects, self.recursion_requested)
    return (patch_bindings_tuples, url_objects)