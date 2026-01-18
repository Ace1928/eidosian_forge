from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReplacementMethodValueValuesEnum(_messages.Enum):
    """What action should be used to replace instances. See
    minimal_action.REPLACE

    Values:
      RECREATE: Instances will be recreated (with the same name)
      SUBSTITUTE: Default option: instances will be deleted and created (with
        a new name)
    """
    RECREATE = 0
    SUBSTITUTE = 1