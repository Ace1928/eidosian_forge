from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryValueValuesEnum(_messages.Enum):
    """Preset option controlling parameters for speed-precision trade-off
    when querying for examples. If omitted, defaults to `PRECISE`.

    Values:
      PRECISE: More precise neighbors as a trade-off against slower response.
      FAST: Faster response as a trade-off against less precise neighbors.
    """
    PRECISE = 0
    FAST = 1