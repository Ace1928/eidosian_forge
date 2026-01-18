from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IngressSource(_messages.Message):
    """The source that IngressPolicy authorizes access from.

  Fields:
    accessLevel: An AccessLevel resource name that allows resources within the
      ServicePerimeters to be accessed from the internet. AccessLevels listed
      must be in the same policy as this ServicePerimeter. Referencing a
      nonexistent AccessLevel will cause an error. If an AccessLevel
      AccessLevel name is not specified, resources within the perimeter can
      only be accessed through Google Cloud calls with request origins within
      the perimeter. Example:
      `accessPolicies/MY_POLICY/accessLevels/MY_LEVEL`. If a single `*` is
      specified for `access_level`, then all IngressSources will be allowed.
    resource: A Google Cloud resource that is allowed to ingress the
      perimeter. Requests from these resources are allowed to access perimeter
      data. Only projects and VPCs are allowed. Project format:
      `projects/{project_number}`. VPC network format: `//compute.googleapis.c
      om/projects/{PROJECT_ID}/global/networks/{NETWORK_NAME}`. The resource
      might be in any Google Cloud organization, not just the organization
      that the perimeter is defined in. `*` is not allowed, the case of
      allowing all Google Cloud resources only is not supported.
  """
    accessLevel = _messages.StringField(1)
    resource = _messages.StringField(2)