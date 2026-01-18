from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsScopesDeleteRequest(_messages.Message):
    """A GkehubProjectsLocationsScopesDeleteRequest object.

  Fields:
    name: Required. The Scope resource name in the format
      `projects/*/locations/*/scopes/*`.
  """
    name = _messages.StringField(1, required=True)