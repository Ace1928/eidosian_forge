from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsRolloutsGetRequest(_messages.Message):
    """A GkehubProjectsLocationsRolloutsGetRequest object.

  Fields:
    name: Required. The name of the rollout to retrieve.
      projects/{project}/locations/{location}/rollouts/{rollout}
  """
    name = _messages.StringField(1, required=True)