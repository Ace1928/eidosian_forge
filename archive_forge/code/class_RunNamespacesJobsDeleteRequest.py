from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunNamespacesJobsDeleteRequest(_messages.Message):
    """A RunNamespacesJobsDeleteRequest object.

  Fields:
    apiVersion: Optional. Cloud Run currently ignores this parameter.
    force: If set to true, the Job and its Executions will be deleted no
      matter whether any Executions are still running or not. If set to false
      or unset, the Job and its Executions can only be deleted if there are no
      running Executions. Any running Execution will fail the deletion.
    kind: Optional. Cloud Run currently ignores this parameter.
    name: Required. The name of the job to delete. Replace {namespace} with
      the project ID or number. It takes the form namespaces/{namespace}. For
      example: namespaces/PROJECT_ID
    propagationPolicy: Optional. Specifies the propagation policy of delete.
      Cloud Run currently ignores this setting, and deletes in the background.
      Please see kubernetes.io/docs/concepts/workloads/controllers/garbage-
      collection/ for more information.
  """
    apiVersion = _messages.StringField(1)
    force = _messages.BooleanField(2)
    kind = _messages.StringField(3)
    name = _messages.StringField(4, required=True)
    propagationPolicy = _messages.StringField(5)