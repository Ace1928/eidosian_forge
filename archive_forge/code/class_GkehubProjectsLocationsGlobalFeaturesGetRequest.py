from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsGlobalFeaturesGetRequest(_messages.Message):
    """A GkehubProjectsLocationsGlobalFeaturesGetRequest object.

  Fields:
    name: Required. The Feature resource name in the format
      `projects/*/locations/global/features/*`
  """
    name = _messages.StringField(1, required=True)