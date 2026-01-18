from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIdentityAccesscontextmanagerV1ServicePerimeterConfig(_messages.Message):
    """`ServicePerimeterConfig` specifies a set of Google Cloud resources that
  describe specific Service Perimeter configuration.

  Fields:
    accessLevels: A list of `AccessLevel` resource names that allow resources
      within the `ServicePerimeter` to be accessed from the internet.
      `AccessLevels` listed must be in the same policy as this
      `ServicePerimeter`. Referencing a nonexistent `AccessLevel` is a syntax
      error. If no `AccessLevel` names are listed, resources within the
      perimeter can only be accessed via Google Cloud calls with request
      origins within the perimeter. Example:
      `"accessPolicies/MY_POLICY/accessLevels/MY_LEVEL"`. For Service
      Perimeter Bridge, must be empty.
    egressPolicies: List of EgressPolicies to apply to the perimeter. A
      perimeter may have multiple EgressPolicies, each of which is evaluated
      separately. Access is granted if any EgressPolicy grants it. Must be
      empty for a perimeter bridge.
    ingressPolicies: List of IngressPolicies to apply to the perimeter. A
      perimeter may have multiple IngressPolicies, each of which is evaluated
      separately. Access is granted if any Ingress Policy grants it. Must be
      empty for a perimeter bridge.
    resources: A list of Google Cloud resources that are inside of the service
      perimeter. Currently only projects and VPCs are allowed. Project format:
      `projects/{project_number}` VPC network format:
      `//compute.googleapis.com/projects/{PROJECT_ID}/global/networks/{NAME}`.
    restrictedServices: Google Cloud services that are subject to the Service
      Perimeter restrictions. For example, if `storage.googleapis.com` is
      specified, access to the storage buckets inside the perimeter must meet
      the perimeter's access restrictions.
    vpcAccessibleServices: Configuration for APIs allowed within Perimeter.
  """
    accessLevels = _messages.StringField(1, repeated=True)
    egressPolicies = _messages.MessageField('GoogleIdentityAccesscontextmanagerV1EgressPolicy', 2, repeated=True)
    ingressPolicies = _messages.MessageField('GoogleIdentityAccesscontextmanagerV1IngressPolicy', 3, repeated=True)
    resources = _messages.StringField(4, repeated=True)
    restrictedServices = _messages.StringField(5, repeated=True)
    vpcAccessibleServices = _messages.MessageField('GoogleIdentityAccesscontextmanagerV1VpcAccessibleServices', 6)