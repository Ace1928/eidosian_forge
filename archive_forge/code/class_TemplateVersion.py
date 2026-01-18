from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TemplateVersion(_messages.Message):
    """////////////////////////////////////////////////////////////////////////
  ///// //// Template Catalog is used to organize user TemplateVersions. ////
  TemplateVersions that have the same project_id and display_name are ////
  belong to the same Template. //// Templates with the same project_id belong
  to the same Project. //// TemplateVersion may have labels and multiple
  labels are allowed. //// Duplicated labels in the same `TemplateVersion` are
  not allowed. //// TemplateVersion may have tags and multiple tags are
  allowed. Duplicated //// tags in the same `Template` are not allowed!

  Enums:
    TypeValueValuesEnum: Either LEGACY or FLEX. This should match with the
      type of artifact.

  Messages:
    LabelsValue: Labels for the Template Version. Labels can be duplicate
      within Template.

  Fields:
    artifact: Job graph and metadata if it is a legacy Template. Container
      image path and metadata if it is flex Template.
    createTime: Creation time of this TemplateVersion.
    description: Template description from the user.
    displayName: A customized name for Template. Multiple TemplateVersions per
      Template.
    labels: Labels for the Template Version. Labels can be duplicate within
      Template.
    projectId: A unique project_id. Multiple Templates per Project.
    tags: Alias for version_id, helps locate a TemplateVersion.
    type: Either LEGACY or FLEX. This should match with the type of artifact.
    versionId: An auto generated version_id for TemplateVersion.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Either LEGACY or FLEX. This should match with the type of artifact.

    Values:
      TEMPLATE_TYPE_UNSPECIFIED: Default value. Not a useful zero case.
      LEGACY: Legacy Template.
      FLEX: Flex Template.
    """
        TEMPLATE_TYPE_UNSPECIFIED = 0
        LEGACY = 1
        FLEX = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels for the Template Version. Labels can be duplicate within
    Template.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    artifact = _messages.MessageField('Artifact', 1)
    createTime = _messages.StringField(2)
    description = _messages.StringField(3)
    displayName = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    projectId = _messages.StringField(6)
    tags = _messages.StringField(7, repeated=True)
    type = _messages.EnumField('TypeValueValuesEnum', 8)
    versionId = _messages.StringField(9)