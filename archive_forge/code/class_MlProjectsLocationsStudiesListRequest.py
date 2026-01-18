from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MlProjectsLocationsStudiesListRequest(_messages.Message):
    """A MlProjectsLocationsStudiesListRequest object.

  Fields:
    parent: Required. The project and location that the study belongs to.
      Format: projects/{project}/locations/{location}
  """
    parent = _messages.StringField(1, required=True)