from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from apitools.base.py import  exceptions as apitools_exc
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import resource
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.generated_clients.apis import apis_map
import six
class API(object):
    """A data holder for returning API data for display."""

    def __init__(self, name, version, is_default, client, base_url):
        self.name = name
        self.version = version
        self.is_default = is_default
        self._client = client
        self.base_url = base_url

    def GetMessagesModule(self):
        return self._client.MESSAGES_MODULE