from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ChangeReport(_messages.Message):
    """Change report associated with a particular service configuration. It
  contains a list of ConfigChanges based on the comparison between two service
  configurations.

  Fields:
    configChanges: List of changes between two service configurations. The
      changes will be alphabetically sorted based on the identifier of each
      change. A ConfigChange identifier is a dot separated path to the
      configuration. Example:
      visibility.rules[selector='LibraryService.CreateBook'].restriction
  """
    configChanges = _messages.MessageField('ConfigChange', 1, repeated=True)