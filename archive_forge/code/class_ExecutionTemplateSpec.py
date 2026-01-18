from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExecutionTemplateSpec(_messages.Message):
    """ExecutionTemplateSpec describes the metadata and spec an Execution
  should have when created from a job.

  Fields:
    metadata: Optional. Optional metadata for this Execution, including labels
      and annotations. The following annotation keys set properties of the
      created execution: * `run.googleapis.com/cloudsql-instances` sets Cloud
      SQL connections. Multiple values should be comma separated. *
      `run.googleapis.com/vpc-access-connector` sets a Serverless VPC Access
      connector. * `run.googleapis.com/vpc-access-egress` sets VPC egress.
      Supported values are `all-traffic`, `all` (deprecated), and `private-
      ranges-only`. `all-traffic` and `all` provide the same functionality.
      `all` is deprecated but will continue to be supported. Prefer `all-
      traffic`.
    spec: Required. ExecutionSpec holds the desired configuration for
      executions of this job.
  """
    metadata = _messages.MessageField('ObjectMeta', 1)
    spec = _messages.MessageField('ExecutionSpec', 2)