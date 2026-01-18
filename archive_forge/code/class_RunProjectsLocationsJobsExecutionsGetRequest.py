from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunProjectsLocationsJobsExecutionsGetRequest(_messages.Message):
    """A RunProjectsLocationsJobsExecutionsGetRequest object.

  Fields:
    name: Required. The full name of the Execution. Format: `projects/{project
      }/locations/{location}/jobs/{job}/executions/{execution}`, where
      `{project}` can be project id or number.
  """
    name = _messages.StringField(1, required=True)