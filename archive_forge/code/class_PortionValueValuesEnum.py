from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PortionValueValuesEnum(_messages.Enum):
    """Portion of this counter, either key or value.

    Values:
      ALL: Counter portion has not been set.
      KEY: Counter reports a key.
      VALUE: Counter reports a value.
    """
    ALL = 0
    KEY = 1
    VALUE = 2