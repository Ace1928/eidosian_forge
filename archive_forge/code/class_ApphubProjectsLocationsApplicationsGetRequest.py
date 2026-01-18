from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApphubProjectsLocationsApplicationsGetRequest(_messages.Message):
    """A ApphubProjectsLocationsApplicationsGetRequest object.

  Fields:
    name: Required. Fully qualified name of the Application to fetch. Expected
      format:
      `projects/{project}/locations/{location}/applications/{application}`.
  """
    name = _messages.StringField(1, required=True)