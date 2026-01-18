from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LicenseCode(_messages.Message):
    """Represents a License Code resource. A License Code is a unique
  identifier used to represent a license resource. *Caution* This resource is
  intended for use only by third-party partners who are creating Cloud
  Marketplace images.

  Enums:
    StateValueValuesEnum: [Output Only] Current state of this License Code.

  Fields:
    creationTimestamp: [Output Only] Creation timestamp in RFC3339 text
      format.
    description: [Output Only] Description of this License Code.
    id: [Output Only] The unique identifier for the resource. This identifier
      is defined by the server.
    kind: [Output Only] Type of resource. Always compute#licenseCode for
      licenses.
    licenseAlias: [Output Only] URL and description aliases of Licenses with
      the same License Code.
    name: [Output Only] Name of the resource. The name is 1-20 characters long
      and must be a valid 64 bit integer.
    selfLink: [Output Only] Server-defined URL for the resource.
    state: [Output Only] Current state of this License Code.
    transferable: [Output Only] If true, the license will remain attached when
      creating images or snapshots from disks. Otherwise, the license is not
      transferred.
  """

    class StateValueValuesEnum(_messages.Enum):
        """[Output Only] Current state of this License Code.

    Values:
      DISABLED: Machines are not allowed to attach boot disks with this
        License Code. Requests to create new resources with this license will
        be rejected.
      ENABLED: Use is allowed for anyone with USE_READ_ONLY access to this
        License Code.
      RESTRICTED: Use of this license is limited to a project whitelist.
      STATE_UNSPECIFIED: <no description>
      TERMINATED: Reserved state.
    """
        DISABLED = 0
        ENABLED = 1
        RESTRICTED = 2
        STATE_UNSPECIFIED = 3
        TERMINATED = 4
    creationTimestamp = _messages.StringField(1)
    description = _messages.StringField(2)
    id = _messages.IntegerField(3, variant=_messages.Variant.UINT64)
    kind = _messages.StringField(4, default='compute#licenseCode')
    licenseAlias = _messages.MessageField('LicenseCodeLicenseAlias', 5, repeated=True)
    name = _messages.StringField(6)
    selfLink = _messages.StringField(7)
    state = _messages.EnumField('StateValueValuesEnum', 8)
    transferable = _messages.BooleanField(9)