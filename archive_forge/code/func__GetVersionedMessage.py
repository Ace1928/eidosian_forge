from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.util import apis
def _GetVersionedMessage(self, message_name):
    """Returns the versioned API messages class by name."""
    return self._GetMessage('{prefix}{name}'.format(prefix=self._message_prefix, name=message_name))