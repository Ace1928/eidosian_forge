from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsGlobalFeaturesCreateRequest(_messages.Message):
    """A GkehubProjectsLocationsGlobalFeaturesCreateRequest object.

  Fields:
    feature: A Feature resource to be passed as the request body.
    featureId: The ID of the feature to create.
    parent: Required. The parent (project and location) where the Feature will
      be created. Specified in the format `projects/*/locations/global`.
  """
    feature = _messages.MessageField('Feature', 1)
    featureId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)