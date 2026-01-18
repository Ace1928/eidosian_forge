from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LeaseTasksRequest(_messages.Message):
    """Request message for leasing tasks using LeaseTasks.

  Enums:
    ResponseViewValueValuesEnum: The response_view specifies which subset of
      the Task will be returned. By default response_view is BASIC; not all
      information is retrieved by default because some data, such as payloads,
      might be desirable to return only when needed because of its large size
      or because of the sensitivity of data that it contains. Authorization
      for FULL requires `cloudtasks.tasks.fullView` [Google
      IAM](https://cloud.google.com/iam/) permission on the Task resource.

  Fields:
    filter: `filter` can be used to specify a subset of tasks to lease. When
      `filter` is set to `tag=` then the response will contain only tasks
      whose tag is equal to ``. `` must be less than 500 characters. When
      `filter` is set to `tag_function=oldest_tag()`, only tasks which have
      the same tag as the task with the oldest schedule_time will be returned.
      Grammar Syntax: * `filter = "tag=" tag | "tag_function=" function` *
      `tag = string` * `function = "oldest_tag()"` The `oldest_tag()` function
      returns tasks which have the same tag as the oldest task (ordered by
      schedule time). SDK compatibility: Although the SDK allows tags to be
      either string or [bytes](https://cloud.google.com/appengine/docs/standar
      d/java/javadoc/com/google/appengine/api/taskqueue/TaskOptions.html#tag-
      byte:A-), only UTF-8 encoded tags can be used in Cloud Tasks. Tag which
      aren't UTF-8 encoded can't be used in the filter and the task's tag will
      be displayed as empty in Cloud Tasks.
    leaseDuration: Required. The duration of the lease. Each task returned in
      the response will have its schedule_time set to the current time plus
      the `lease_duration`. The task is leased until its schedule_time; thus,
      the task will not be returned to another LeaseTasks call before its
      schedule_time. After the worker has successfully finished the work
      associated with the task, the worker must call via AcknowledgeTask
      before the schedule_time. Otherwise the task will be returned to a later
      LeaseTasks call so that another worker can retry it. The maximum lease
      duration is 1 week. `lease_duration` will be truncated to the nearest
      second.
    maxTasks: The maximum number of tasks to lease. The system will make a
      best effort to return as close to as `max_tasks` as possible. The
      largest that `max_tasks` can be is 1000. The maximum total size of a
      lease tasks response is 32 MB. If the sum of all task sizes requested
      reaches this limit, fewer tasks than requested are returned.
    responseView: The response_view specifies which subset of the Task will be
      returned. By default response_view is BASIC; not all information is
      retrieved by default because some data, such as payloads, might be
      desirable to return only when needed because of its large size or
      because of the sensitivity of data that it contains. Authorization for
      FULL requires `cloudtasks.tasks.fullView` [Google
      IAM](https://cloud.google.com/iam/) permission on the Task resource.
  """

    class ResponseViewValueValuesEnum(_messages.Enum):
        """The response_view specifies which subset of the Task will be returned.
    By default response_view is BASIC; not all information is retrieved by
    default because some data, such as payloads, might be desirable to return
    only when needed because of its large size or because of the sensitivity
    of data that it contains. Authorization for FULL requires
    `cloudtasks.tasks.fullView` [Google IAM](https://cloud.google.com/iam/)
    permission on the Task resource.

    Values:
      VIEW_UNSPECIFIED: Unspecified. Defaults to BASIC.
      BASIC: The basic view omits fields which can be large or can contain
        sensitive data. This view does not include the (payload in
        AppEngineHttpRequest and payload in PullMessage). These payloads are
        desirable to return only when needed, because they can be large and
        because of the sensitivity of the data that you choose to store in it.
      FULL: All information is returned. Authorization for FULL requires
        `cloudtasks.tasks.fullView` [Google
        IAM](https://cloud.google.com/iam/) permission on the Queue resource.
    """
        VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2
    filter = _messages.StringField(1)
    leaseDuration = _messages.StringField(2)
    maxTasks = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    responseView = _messages.EnumField('ResponseViewValueValuesEnum', 4)