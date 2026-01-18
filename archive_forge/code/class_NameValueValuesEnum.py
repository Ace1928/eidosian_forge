from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NameValueValuesEnum(_messages.Enum):
    """The name of partner data collector party collecting the asset.

    Values:
      DATA_COLLECTOR_UNSPECIFIED: The data collector is unspecified.
      ATTACK_PATH_SIMULATION: <no description>
    """
    DATA_COLLECTOR_UNSPECIFIED = 0
    ATTACK_PATH_SIMULATION = 1