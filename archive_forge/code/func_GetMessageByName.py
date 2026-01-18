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
def GetMessageByName(self, name):
    """Gets a arbitrary apitools message class by name.

    This method can be used to get arbitrary apitools messages from the
    underlying service. Examples:

    policy_type = method.GetMessageByName('Policy')
    status_type = method.GetMessageByName('Status')

    Args:
      name: str, the name of the message to return.
    Returns:
      The apitools Message object.
    """
    msgs = self._service.client.MESSAGES_MODULE
    return getattr(msgs, name, None)