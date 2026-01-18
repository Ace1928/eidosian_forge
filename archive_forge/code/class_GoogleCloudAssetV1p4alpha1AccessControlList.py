from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudAssetV1p4alpha1AccessControlList(_messages.Message):
    """An access control list, derived from the above IAM policy binding, which
  contains a set of resources and accesses. May include one item from each set
  to compose an access control entry.  NOTICE that there could be multiple
  access control lists for one IAM policy binding. The access control lists
  are created per resource type.  For example, assume we have the following
  cases in one IAM policy binding: - Permission P1 and P2 apply to resource R1
  and R2 of resource type RT1; - Permission P3 applies to resource R3 and R4
  of resource type RT2;  This will result in the following access control
  lists: - AccessControlList 1: RT1, [R1, R2], [P1, P2] - AccessControlList 2:
  RT2, [R3, R4], [P3]

  Fields:
    accesses: The accesses that match one of the following conditions: - The
      access_selector, if it is specified in request; - Otherwise, access
      specifiers reachable from the policy binding's role.
    baseResourceType: The unified resource type name of the resource type that
      this access control list is based on, such as
      "compute.googleapis.com/Instance" for Compute Engine instance, etc.
    resourceEdges: Resource edges of the graph starting from the policy
      attached resource to any descendant resources. The Edge.source_node
      contains the full resource name of a parent resource and
      Edge.target_node contains the full resource name of a child resource.
      This field is present only if the output_resource_edges option is
      enabled in request.
    resources: The resources that match one of the following conditions: - The
      resource_selector, if it is specified in request; - Otherwise, resources
      reachable from the policy attached resource.
  """
    accesses = _messages.MessageField('GoogleCloudAssetV1p4alpha1Access', 1, repeated=True)
    baseResourceType = _messages.StringField(2)
    resourceEdges = _messages.MessageField('GoogleCloudAssetV1p4alpha1Edge', 3, repeated=True)
    resources = _messages.MessageField('GoogleCloudAssetV1p4alpha1Resource', 4, repeated=True)