from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicysimulatorV1alphaOrgPolicyViolationsPreview(_messages.Message):
    """OrgPolicyViolationsPreview is a resource providing a preview of the
  violations that will exist if an OrgPolicy change is made. The list of
  violations are modeled as child resources and retrieved via a
  ListOrgPolicyViolations API call. There are potentially more
  OrgPolicyViolations than could fit in an embedded field. Thus, the use of a
  child resource instead of a field.

  Enums:
    StateValueValuesEnum: Output only. The state of the
      `OrgPolicyViolationsPreview`.

  Fields:
    createTime: Output only. Time when this `OrgPolicyViolationsPreview` was
      created.
    customConstraints: Output only. The names of the constraints against which
      all `OrgPolicyViolations` were evaluated. If `OrgPolicyOverlay` only
      contains `PolicyOverlay` then it contains the name of the configured
      custom constraint, applicable to the specified policies. Otherwise it
      contains the name of the constraint specified in
      `CustomConstraintOverlay`. Format: `organizations/{organization_id}/cust
      omConstraints/{custom_constraint_id}` Example:
      `organizations/123/customConstraints/custom.createOnlyE2TypeVms`
    name: Output only. The resource name of the `OrgPolicyViolationsPreview`.
      It has the following format: `organizations/{organization}/locations/{lo
      cation}/orgPolicyViolationsPreviews/{orgPolicyViolationsPreview}`
      Example: `organizations/my-example-
      org/locations/global/orgPolicyViolationsPreviews/506a5f7f`
    overlay: Required. The proposed changes we are previewing violations for.
    resourceCounts: Output only. A summary of the state of all resources
      scanned for compliance with the changed OrgPolicy.
    state: Output only. The state of the `OrgPolicyViolationsPreview`.
    violationsCount: Output only. The number of OrgPolicyViolations in this
      `OrgPolicyViolationsPreview`. This count may differ from
      `resource_summary.noncompliant_count` because each OrgPolicyViolation is
      specific to a resource **and** constraint. If there are multiple
      constraints being evaluated (i.e. multiple policies in the overlay), a
      single resource may violate multiple constraints.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the `OrgPolicyViolationsPreview`.

    Values:
      PREVIEW_STATE_UNSPECIFIED: The state is unspecified.
      PREVIEW_PENDING: The OrgPolicyViolationsPreview has not been created
        yet.
      PREVIEW_RUNNING: The OrgPolicyViolationsPreview is currently being
        created.
      PREVIEW_SUCCEEDED: The OrgPolicyViolationsPreview creation finished
        successfully.
      PREVIEW_FAILED: The OrgPolicyViolationsPreview creation failed with an
        error.
    """
        PREVIEW_STATE_UNSPECIFIED = 0
        PREVIEW_PENDING = 1
        PREVIEW_RUNNING = 2
        PREVIEW_SUCCEEDED = 3
        PREVIEW_FAILED = 4
    createTime = _messages.StringField(1)
    customConstraints = _messages.StringField(2, repeated=True)
    name = _messages.StringField(3)
    overlay = _messages.MessageField('GoogleCloudPolicysimulatorV1alphaOrgPolicyOverlay', 4)
    resourceCounts = _messages.MessageField('GoogleCloudPolicysimulatorV1alphaOrgPolicyViolationsPreviewResourceCounts', 5)
    state = _messages.EnumField('StateValueValuesEnum', 6)
    violationsCount = _messages.IntegerField(7, variant=_messages.Variant.INT32)