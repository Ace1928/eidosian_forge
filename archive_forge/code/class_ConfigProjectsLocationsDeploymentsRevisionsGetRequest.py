from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigProjectsLocationsDeploymentsRevisionsGetRequest(_messages.Message):
    """A ConfigProjectsLocationsDeploymentsRevisionsGetRequest object.

  Fields:
    name: Required. The name of the Revision in the format: 'projects/{project
      _id}/locations/{location}/deployments/{deployment}/revisions/{revision}'
      .
  """
    name = _messages.StringField(1, required=True)