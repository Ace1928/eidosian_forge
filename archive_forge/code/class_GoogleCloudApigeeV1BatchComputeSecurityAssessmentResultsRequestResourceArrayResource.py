from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1BatchComputeSecurityAssessmentResultsRequestResourceArrayResource(_messages.Message):
    """Resource for which we are computing security assessment.

  Enums:
    TypeValueValuesEnum: Required. Type of this resource.

  Fields:
    name: Required. Name of this resource.
    type: Required. Type of this resource.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Required. Type of this resource.

    Values:
      RESOURCE_TYPE_UNSPECIFIED: ResourceType not specified.
      API_PROXY: Resource is an Apigee Proxy.
    """
        RESOURCE_TYPE_UNSPECIFIED = 0
        API_PROXY = 1
    name = _messages.StringField(1)
    type = _messages.EnumField('TypeValueValuesEnum', 2)