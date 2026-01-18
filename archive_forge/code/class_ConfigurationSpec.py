from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigurationSpec(_messages.Message):
    """ConfigurationSpec holds the desired state of the Configuration (from the
  client).

  Fields:
    template: Template holds the latest specification for the Revision to be
      stamped out.
  """
    template = _messages.MessageField('RevisionTemplate', 1)