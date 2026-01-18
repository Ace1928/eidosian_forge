from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunNamespacesTasksListRequest(_messages.Message):
    """A RunNamespacesTasksListRequest object.

  Fields:
    continue_: Optional. Optional encoded string to continue paging.
    fieldSelector: Optional. Not supported by Cloud Run.
    includeUninitialized: Optional. Not supported by Cloud Run.
    labelSelector: Optional. Allows to filter resources based on a label.
      Supported operations are =, !=, exists, in, and notIn. For example, to
      list all tasks of execution "foo" in succeeded state: `run.googleapis.co
      m/execution=foo,run.googleapis.com/runningState=Succeeded`. Supported
      states are: * `Pending`: Initial state of all tasks. The task has not
      yet started but eventually will. * `Running`: Container instances for
      this task are running or will be running shortly. * `Succeeded`: No more
      container instances to run for the task, and the last attempt succeeded.
      * `Failed`: No more container instances to run for the task, and the
      last attempt failed. This task has run out of retry attempts. *
      `Cancelled`: Task was running but got stopped because its parent
      execution has been aborted. * `Abandoned`: The task has not yet started
      and never will because its parent execution has been aborted.
    limit: Optional. The maximum number of records that should be returned.
    parent: Required. The namespace from which the tasks should be listed.
      Replace {namespace} with the project ID or number. It takes the form
      namespaces/{namespace}. For example: namespaces/PROJECT_ID
    resourceVersion: Optional. Not supported by Cloud Run.
    watch: Optional. Not supported by Cloud Run.
  """
    continue_ = _messages.StringField(1)
    fieldSelector = _messages.StringField(2)
    includeUninitialized = _messages.BooleanField(3)
    labelSelector = _messages.StringField(4)
    limit = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    parent = _messages.StringField(6, required=True)
    resourceVersion = _messages.StringField(7)
    watch = _messages.BooleanField(8)