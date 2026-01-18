from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.core import yaml
def _GetMessage(self, message_name):
    """Returns the API messsages class by name."""
    return getattr(self._messages, '{prefix}{name}'.format(prefix=self._message_prefix, name=message_name), None)