from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsMembershipsValidateExclusivityRequest(_messages.Message):
    """A GkehubProjectsLocationsMembershipsValidateExclusivityRequest object.

  Fields:
    crManifest: Optional. The YAML of the membership CR in the cluster. Empty
      if the membership CR does not exist.
    intendedMembership: Required. The intended membership name under the
      `parent`. This method only does validation in anticipation of a
      CreateMembership call with the same name.
    parent: Required. The parent (project and location) where the Memberships
      will be created. Specified in the format `projects/*/locations/*`.
  """
    crManifest = _messages.StringField(1)
    intendedMembership = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)