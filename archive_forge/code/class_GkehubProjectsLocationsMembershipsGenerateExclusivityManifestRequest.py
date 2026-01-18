from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsMembershipsGenerateExclusivityManifestRequest(_messages.Message):
    """A GkehubProjectsLocationsMembershipsGenerateExclusivityManifestRequest
  object.

  Fields:
    crManifest: Optional. The YAML manifest of the membership CR retrieved by
      `kubectl get memberships membership`. Leave empty if the resource does
      not exist.
    crdManifest: Optional. The YAML manifest of the membership CRD retrieved
      by `kubectl get customresourcedefinitions membership`. Leave empty if
      the resource does not exist.
    name: Required. The Membership resource name in the format
      `projects/*/locations/*/memberships/*`.
  """
    crManifest = _messages.StringField(1)
    crdManifest = _messages.StringField(2)
    name = _messages.StringField(3, required=True)