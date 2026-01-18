from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsSessionsSparkApplicationsAccessJobRequest(_messages.Message):
    """A DataprocProjectsLocationsSessionsSparkApplicationsAccessJobRequest
  object.

  Fields:
    jobId: Required. Job ID to fetch data for.
    name: Required. The fully qualified name of the session to retrieve in the
      format "projects/PROJECT_ID/locations/DATAPROC_REGION/sessions/SESSION_I
      D/sparkApplications/APPLICATION_ID"
    parent: Required. Parent (Session) resource reference.
  """
    jobId = _messages.IntegerField(1)
    name = _messages.StringField(2, required=True)
    parent = _messages.StringField(3)