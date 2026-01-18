from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AptSettings(_messages.Message):
    """Apt patching is completed by executing `apt-get update && apt-get
  upgrade`. Additional options can be set to control how this is executed.

  Enums:
    TypeValueValuesEnum: By changing the type to DIST, the patching is
      performed using `apt-get dist-upgrade` instead.

  Fields:
    excludes: List of packages to exclude from update. These packages will be
      excluded
    exclusivePackages: An exclusive list of packages to be updated. These are
      the only packages that will be updated. If these packages are not
      installed, they will be ignored. This field cannot be specified with any
      other patch configuration fields.
    type: By changing the type to DIST, the patching is performed using `apt-
      get dist-upgrade` instead.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """By changing the type to DIST, the patching is performed using `apt-get
    dist-upgrade` instead.

    Values:
      TYPE_UNSPECIFIED: By default, upgrade will be performed.
      DIST: Runs `apt-get dist-upgrade`.
      UPGRADE: Runs `apt-get upgrade`.
    """
        TYPE_UNSPECIFIED = 0
        DIST = 1
        UPGRADE = 2
    excludes = _messages.StringField(1, repeated=True)
    exclusivePackages = _messages.StringField(2, repeated=True)
    type = _messages.EnumField('TypeValueValuesEnum', 3)