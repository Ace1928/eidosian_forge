from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsMembershipsGetRequest(_messages.Message):
    """A GkehubProjectsLocationsMembershipsGetRequest object.

  Fields:
    name: Required. The Membership resource name in the format
      `projects/*/locations/*/memberships/*`.
  """
    name = _messages.StringField(1, required=True)