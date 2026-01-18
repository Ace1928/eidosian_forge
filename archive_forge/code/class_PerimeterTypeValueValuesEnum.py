from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PerimeterTypeValueValuesEnum(_messages.Enum):
    """Perimeter type indicator. A single project or VPC network is allowed
    to be a member of single regular perimeter, but multiple service perimeter
    bridges. A project cannot be a included in a perimeter bridge without
    being included in regular perimeter. For perimeter bridges, the restricted
    service list as well as access level lists must be empty.

    Values:
      PERIMETER_TYPE_REGULAR: Regular Perimeter. When no value is specified,
        the perimeter uses this type.
      PERIMETER_TYPE_BRIDGE: Perimeter Bridge.
    """
    PERIMETER_TYPE_REGULAR = 0
    PERIMETER_TYPE_BRIDGE = 1