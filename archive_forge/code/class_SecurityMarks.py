from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityMarks(_messages.Message):
    """User specified security marks that are attached to the parent Security
  Command Center resource. Security marks are scoped within a Security Command
  Center organization -- they can be modified and viewed by all users who have
  proper permissions on the organization.

  Messages:
    MarksValue: Mutable user specified security marks belonging to the parent
      resource. Constraints are as follows: * Keys and values are treated as
      case insensitive * Keys must be between 1 - 256 characters (inclusive) *
      Keys must be letters, numbers, underscores, or dashes * Values have
      leading and trailing whitespace trimmed, remaining characters must be
      between 1 - 4096 characters (inclusive)

  Fields:
    canonicalName: The canonical name of the marks. Examples:
      "organizations/{organization_id}/assets/{asset_id}/securityMarks"
      "folders/{folder_id}/assets/{asset_id}/securityMarks"
      "projects/{project_number}/assets/{asset_id}/securityMarks" "organizatio
      ns/{organization_id}/sources/{source_id}/findings/{finding_id}/securityM
      arks" "folders/{folder_id}/sources/{source_id}/findings/{finding_id}/sec
      urityMarks" "projects/{project_number}/sources/{source_id}/findings/{fin
      ding_id}/securityMarks"
    marks: Mutable user specified security marks belonging to the parent
      resource. Constraints are as follows: * Keys and values are treated as
      case insensitive * Keys must be between 1 - 256 characters (inclusive) *
      Keys must be letters, numbers, underscores, or dashes * Values have
      leading and trailing whitespace trimmed, remaining characters must be
      between 1 - 4096 characters (inclusive)
    name: The relative resource name of the SecurityMarks. See:
      https://cloud.google.com/apis/design/resource_names#relative_resource_na
      me Examples:
      "organizations/{organization_id}/assets/{asset_id}/securityMarks" "organ
      izations/{organization_id}/sources/{source_id}/findings/{finding_id}/sec
      urityMarks".
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MarksValue(_messages.Message):
        """Mutable user specified security marks belonging to the parent
    resource. Constraints are as follows: * Keys and values are treated as
    case insensitive * Keys must be between 1 - 256 characters (inclusive) *
    Keys must be letters, numbers, underscores, or dashes * Values have
    leading and trailing whitespace trimmed, remaining characters must be
    between 1 - 4096 characters (inclusive)

    Messages:
      AdditionalProperty: An additional property for a MarksValue object.

    Fields:
      additionalProperties: Additional properties of type MarksValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MarksValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    canonicalName = _messages.StringField(1)
    marks = _messages.MessageField('MarksValue', 2)
    name = _messages.StringField(3)