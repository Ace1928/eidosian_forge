from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GetTemplateResponse(_messages.Message):
    """The response to a GetTemplate request.

  Enums:
    TemplateTypeValueValuesEnum: Template Type.

  Fields:
    metadata: The template metadata describing the template name, available
      parameters, etc.
    runtimeMetadata: Describes the runtime metadata with SDKInfo and available
      parameters.
    status: The status of the get template request. Any problems with the
      request will be indicated in the error_details.
    templateType: Template Type.
  """

    class TemplateTypeValueValuesEnum(_messages.Enum):
        """Template Type.

    Values:
      UNKNOWN: Unknown Template Type.
      LEGACY: Legacy Template.
      FLEX: Flex Template.
    """
        UNKNOWN = 0
        LEGACY = 1
        FLEX = 2
    metadata = _messages.MessageField('TemplateMetadata', 1)
    runtimeMetadata = _messages.MessageField('RuntimeMetadata', 2)
    status = _messages.MessageField('Status', 3)
    templateType = _messages.EnumField('TemplateTypeValueValuesEnum', 4)