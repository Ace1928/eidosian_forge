from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import folder_util
from googlecloudsdk.command_lib.storage import plurality_checkable_iterator
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import log
from googlecloudsdk.core.util import debug_output
class BucketSetting(enum.Enum):
    """An enum saying whether or not to include matched buckets."""
    YES = '_yes'
    NO = '_no'
    NO_WITH_ERROR = '_no_with_error'