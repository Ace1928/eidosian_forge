from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ModifyTemplateVersionLabelRequest(_messages.Message):
    """Either add the label to TemplateVersion or remove it from the
  TemplateVersion.

  Enums:
    OpValueValuesEnum: Requests for add label to TemplateVersion or remove
      label from TemplateVersion.

  Fields:
    key: The label key for update.
    op: Requests for add label to TemplateVersion or remove label from
      TemplateVersion.
    value: The label value for update.
  """

    class OpValueValuesEnum(_messages.Enum):
        """Requests for add label to TemplateVersion or remove label from
    TemplateVersion.

    Values:
      OPERATION_UNSPECIFIED: Default value.
      ADD: Add the label to the TemplateVersion object.
      REMOVE: Remove the label from the TemplateVersion object.
    """
        OPERATION_UNSPECIFIED = 0
        ADD = 1
        REMOVE = 2
    key = _messages.StringField(1)
    op = _messages.EnumField('OpValueValuesEnum', 2)
    value = _messages.StringField(3)