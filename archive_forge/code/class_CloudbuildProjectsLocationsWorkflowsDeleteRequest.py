from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsWorkflowsDeleteRequest(_messages.Message):
    """A CloudbuildProjectsLocationsWorkflowsDeleteRequest object.

  Fields:
    etag: The etag of the workflow. If this is provided, it must match the
      server's etag.
    name: Required. Format:
      `projects/{project}/locations/{location}/workflow/{workflow}`
    validateOnly: When true, the query is validated only, but not executed.
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    validateOnly = _messages.BooleanField(3)