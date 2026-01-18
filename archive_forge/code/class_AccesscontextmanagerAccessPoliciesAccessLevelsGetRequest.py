from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccesscontextmanagerAccessPoliciesAccessLevelsGetRequest(_messages.Message):
    """A AccesscontextmanagerAccessPoliciesAccessLevelsGetRequest object.

  Enums:
    AccessLevelFormatValueValuesEnum: Whether to return `BasicLevels` in the
      Cloud Common Expression Language rather than as `BasicLevels`. Defaults
      to AS_DEFINED, where Access Levels are returned as `BasicLevels` or
      `CustomLevels` based on how they were created. If set to CEL, all Access
      Levels are returned as `CustomLevels`. In the CEL case, `BasicLevels`
      are translated to equivalent `CustomLevels`.

  Fields:
    accessLevelFormat: Whether to return `BasicLevels` in the Cloud Common
      Expression Language rather than as `BasicLevels`. Defaults to
      AS_DEFINED, where Access Levels are returned as `BasicLevels` or
      `CustomLevels` based on how they were created. If set to CEL, all Access
      Levels are returned as `CustomLevels`. In the CEL case, `BasicLevels`
      are translated to equivalent `CustomLevels`.
    name: Required. Resource name for the Access Level. Format:
      `accessPolicies/{policy_id}/accessLevels/{access_level_id}`
  """

    class AccessLevelFormatValueValuesEnum(_messages.Enum):
        """Whether to return `BasicLevels` in the Cloud Common Expression
    Language rather than as `BasicLevels`. Defaults to AS_DEFINED, where
    Access Levels are returned as `BasicLevels` or `CustomLevels` based on how
    they were created. If set to CEL, all Access Levels are returned as
    `CustomLevels`. In the CEL case, `BasicLevels` are translated to
    equivalent `CustomLevels`.

    Values:
      LEVEL_FORMAT_UNSPECIFIED: The format was not specified.
      AS_DEFINED: Uses the format the resource was defined in. BasicLevels are
        returned as BasicLevels, CustomLevels are returned as CustomLevels.
      CEL: Use Cloud Common Expression Language when returning the resource.
        Both BasicLevels and CustomLevels are returned as CustomLevels.
    """
        LEVEL_FORMAT_UNSPECIFIED = 0
        AS_DEFINED = 1
        CEL = 2
    accessLevelFormat = _messages.EnumField('AccessLevelFormatValueValuesEnum', 1)
    name = _messages.StringField(2, required=True)