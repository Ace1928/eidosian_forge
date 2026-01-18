from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import collections
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.run import condition
from googlecloudsdk.core.console import console_attr
import six
def InitializedInstance(msg_cls):
    """Produce an instance of msg_cls, with all sub-messages initialized.

  Args:
    msg_cls: A message-class to be instantiated.

  Returns:
    An instance of the given class, with all fields initialized blank objects.
  """

    def Instance(field):
        if field.repeated:
            return []
        return InitializedInstance(field.message_type)

    def IncludeField(field):
        return isinstance(field, messages.MessageField)
    args = {field.name: Instance(field) for field in msg_cls.all_fields() if IncludeField(field)}
    return msg_cls(**args)