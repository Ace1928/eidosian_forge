from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobCreationReason(_messages.Message):
    """Reason about why a Job was created from a [`jobs.query`](https://cloud.g
  oogle.com/bigquery/docs/reference/rest/v2/jobs/query) method when used with
  `JOB_CREATION_OPTIONAL` Job creation mode. For [`jobs.insert`](https://cloud
  .google.com/bigquery/docs/reference/rest/v2/jobs/insert) method calls it
  will always be `REQUESTED`. This feature is not yet available. Jobs will
  always be created.

  Enums:
    CodeValueValuesEnum: Output only. Specifies the high level reason why a
      Job was created.

  Fields:
    code: Output only. Specifies the high level reason why a Job was created.
  """

    class CodeValueValuesEnum(_messages.Enum):
        """Output only. Specifies the high level reason why a Job was created.

    Values:
      CODE_UNSPECIFIED: Reason is not specified.
      REQUESTED: Job creation was requested.
      LONG_RUNNING: The query request ran beyond a system defined timeout
        specified by the [timeoutMs field in the QueryRequest](https://cloud.g
        oogle.com/bigquery/docs/reference/rest/v2/jobs/query#queryrequest). As
        a result it was considered a long running operation for which a job
        was created.
      LARGE_RESULTS: The results from the query cannot fit in the response.
      OTHER: BigQuery has determined that the query needs to be executed as a
        Job.
    """
        CODE_UNSPECIFIED = 0
        REQUESTED = 1
        LONG_RUNNING = 2
        LARGE_RESULTS = 3
        OTHER = 4
    code = _messages.EnumField('CodeValueValuesEnum', 1)