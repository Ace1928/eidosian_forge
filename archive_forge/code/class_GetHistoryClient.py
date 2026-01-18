from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.asset import utils as asset_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
class GetHistoryClient(object):
    """Client for get history assets."""

    def __init__(self, api_version=DEFAULT_API_VERSION):
        self.api_version = api_version
        self.client = GetClient(api_version)
        self.service = self.client.v1

    def GetHistory(self, args):
        return MakeGetAssetsHistoryHttpRequests(args, self.service, self.api_version)