from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsFleetsCreateRequest(_messages.Message):
    """A GkehubProjectsLocationsFleetsCreateRequest object.

  Fields:
    fleet: A Fleet resource to be passed as the request body.
    parent: Required. The parent (project and location) where the Fleet will
      be created. Specified in the format `projects/*/locations/*`.
  """
    fleet = _messages.MessageField('Fleet', 1)
    parent = _messages.StringField(2, required=True)