from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsWorkflowsGetRequest(_messages.Message):
    """A CloudbuildProjectsLocationsWorkflowsGetRequest object.

  Fields:
    name: Required. Format:
      `projects/{project}/locations/{location}/workflow/{workflow}`
  """
    name = _messages.StringField(1, required=True)