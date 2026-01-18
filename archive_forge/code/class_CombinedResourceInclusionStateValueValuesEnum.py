from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CombinedResourceInclusionStateValueValuesEnum(_messages.Enum):
    """Output only. Indicates whether any resource of the rule is the
    specified resource or includes the specified resource.

    Values:
      RESOURCE_INCLUSION_STATE_UNSPECIFIED: An error occurred when checking
        whether the resource includes the specified resource.
      RESOURCE_INCLUSION_STATE_INCLUDED: The resource includes the specified
        resource.
      RESOURCE_INCLUSION_STATE_NOT_INCLUDED: The resource doesn't include the
        specified resource.
      RESOURCE_INCLUSION_STATE_UNKNOWN_INFO: The sender of the request does
        not have access to the relevant data to check whether the resource
        includes the specified resource.
      RESOURCE_INCLUSION_STATE_UNKNOWN_UNSUPPORTED: The resource is of an
        unsupported type, such as non-CRM resources.
    """
    RESOURCE_INCLUSION_STATE_UNSPECIFIED = 0
    RESOURCE_INCLUSION_STATE_INCLUDED = 1
    RESOURCE_INCLUSION_STATE_NOT_INCLUDED = 2
    RESOURCE_INCLUSION_STATE_UNKNOWN_INFO = 3
    RESOURCE_INCLUSION_STATE_UNKNOWN_UNSUPPORTED = 4