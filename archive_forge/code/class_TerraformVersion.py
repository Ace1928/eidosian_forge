from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TerraformVersion(_messages.Message):
    """A TerraformVersion represents the support state the corresponding
  Terraform version.

  Enums:
    StateValueValuesEnum: Output only. The state of the version, ACTIVE,
      DEPRECATED or OBSOLETE.

  Fields:
    deprecateTime: Output only. When the version is deprecated.
    name: Identifier. The version name is in the format: 'projects/{project_id
      }/locations/{location}/terraformVersions/{terraform_version}'.
    obsoleteTime: Output only. When the version is obsolete.
    state: Output only. The state of the version, ACTIVE, DEPRECATED or
      OBSOLETE.
    supportTime: Output only. When the version is supported.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the version, ACTIVE, DEPRECATED or OBSOLETE.

    Values:
      STATE_UNSPECIFIED: The default value. This value is used if the state is
        omitted.
      ACTIVE: The version is actively supported.
      DEPRECATED: The version is deprecated.
      OBSOLETE: The version is obsolete.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        DEPRECATED = 2
        OBSOLETE = 3
    deprecateTime = _messages.StringField(1)
    name = _messages.StringField(2)
    obsoleteTime = _messages.StringField(3)
    state = _messages.EnumField('StateValueValuesEnum', 4)
    supportTime = _messages.StringField(5)