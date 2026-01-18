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
def GetIamHelper(self, storage_url, thread_state=None):
    """Gets an IAM policy for a single, resolved bucket / object URL.

    Args:
      storage_url: A CloudUrl instance with no wildcards, pointing to a
                   specific bucket or object.
      thread_state: CloudApiDelegator instance which is passed from
                    command.WorkerThread.__init__() if the global -m flag is
                    specified. Will use self.gsutil_api if thread_state is set
                    to None.

    Returns:
      Policy instance.
    """
    gsutil_api = GetCloudApiInstance(self, thread_state=thread_state)
    if storage_url.IsBucket():
        policy = gsutil_api.GetBucketIamPolicy(storage_url.bucket_name, provider=storage_url.scheme, fields=['bindings', 'etag'])
    else:
        policy = gsutil_api.GetObjectIamPolicy(storage_url.bucket_name, storage_url.object_name, generation=storage_url.generation, provider=storage_url.scheme, fields=['bindings', 'etag'])
    return policy