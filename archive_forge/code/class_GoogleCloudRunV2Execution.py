from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2Execution(_messages.Message):
    """Execution represents the configuration of a single execution. A
  execution an immutable resource that references a container image which is
  run to completion.

  Enums:
    LaunchStageValueValuesEnum: The least stable launch stage needed to create
      this resource, as defined by [Google Cloud Platform Launch
      Stages](https://cloud.google.com/terms/launch-stages). Cloud Run
      supports `ALPHA`, `BETA`, and `GA`. Note that this value might not be
      what was used as input. For example, if ALPHA was provided as input in
      the parent resource, but only BETA and GA-level features are were, this
      field will be BETA.

  Messages:
    AnnotationsValue: Output only. Unstructured key value map that may be set
      by external tools to store and arbitrary metadata. They are not
      queryable and should be preserved when modifying objects.
    LabelsValue: Output only. Unstructured key value map that can be used to
      organize and categorize objects. User-provided labels are shared with
      Google's billing system, so they can be used to filter, or break down
      billing charges by team, component, environment, state, etc. For more
      information, visit https://cloud.google.com/resource-
      manager/docs/creating-managing-labels or
      https://cloud.google.com/run/docs/configuring/labels

  Fields:
    annotations: Output only. Unstructured key value map that may be set by
      external tools to store and arbitrary metadata. They are not queryable
      and should be preserved when modifying objects.
    cancelledCount: Output only. The number of tasks which reached phase
      Cancelled.
    completionTime: Output only. Represents time when the execution was
      completed. It is not guaranteed to be set in happens-before order across
      separate operations.
    conditions: Output only. The Condition of this Execution, containing its
      readiness status, and detailed error information in case it did not
      reach the desired state.
    createTime: Output only. Represents time when the execution was
      acknowledged by the execution controller. It is not guaranteed to be set
      in happens-before order across separate operations.
    deleteTime: Output only. For a deleted resource, the deletion time. It is
      only populated as a response to a Delete request.
    etag: Output only. A system-generated fingerprint for this version of the
      resource. May be used to detect modification conflict during updates.
    expireTime: Output only. For a deleted resource, the time after which it
      will be permamently deleted. It is only populated as a response to a
      Delete request.
    failedCount: Output only. The number of tasks which reached phase Failed.
    generation: Output only. A number that monotonically increases every time
      the user modifies the desired state.
    job: Output only. The name of the parent Job.
    labels: Output only. Unstructured key value map that can be used to
      organize and categorize objects. User-provided labels are shared with
      Google's billing system, so they can be used to filter, or break down
      billing charges by team, component, environment, state, etc. For more
      information, visit https://cloud.google.com/resource-
      manager/docs/creating-managing-labels or
      https://cloud.google.com/run/docs/configuring/labels
    launchStage: The least stable launch stage needed to create this resource,
      as defined by [Google Cloud Platform Launch
      Stages](https://cloud.google.com/terms/launch-stages). Cloud Run
      supports `ALPHA`, `BETA`, and `GA`. Note that this value might not be
      what was used as input. For example, if ALPHA was provided as input in
      the parent resource, but only BETA and GA-level features are were, this
      field will be BETA.
    logUri: Output only. URI where logs for this execution can be found in
      Cloud Console.
    name: Output only. The unique name of this Execution.
    observedGeneration: Output only. The generation of this Execution. See
      comments in `reconciling` for additional information on reconciliation
      process in Cloud Run.
    parallelism: Output only. Specifies the maximum desired number of tasks
      the execution should run at any given time. Must be <= task_count. The
      actual number of tasks running in steady state will be less than this
      number when ((.spec.task_count - .status.successful) <
      .spec.parallelism), i.e. when the work left to do is less than max
      parallelism.
    reconciling: Output only. Indicates whether the resource's reconciliation
      is still in progress. See comments in `Job.reconciling` for additional
      information on reconciliation process in Cloud Run.
    retriedCount: Output only. The number of tasks which have retried at least
      once.
    runningCount: Output only. The number of actively running tasks.
    satisfiesPzs: Output only. Reserved for future use.
    startTime: Output only. Represents time when the execution started to run.
      It is not guaranteed to be set in happens-before order across separate
      operations.
    succeededCount: Output only. The number of tasks which reached phase
      Succeeded.
    taskCount: Output only. Specifies the desired number of tasks the
      execution should run. Setting to 1 means that parallelism is limited to
      1 and the success of that task signals the success of the execution.
    template: Output only. The template used to create tasks for this
      execution.
    uid: Output only. Server assigned unique identifier for the Execution. The
      value is a UUID4 string and guaranteed to remain unchanged until the
      resource is deleted.
    updateTime: Output only. The last-modified time.
  """

    class LaunchStageValueValuesEnum(_messages.Enum):
        """The least stable launch stage needed to create this resource, as
    defined by [Google Cloud Platform Launch
    Stages](https://cloud.google.com/terms/launch-stages). Cloud Run supports
    `ALPHA`, `BETA`, and `GA`. Note that this value might not be what was used
    as input. For example, if ALPHA was provided as input in the parent
    resource, but only BETA and GA-level features are were, this field will be
    BETA.

    Values:
      LAUNCH_STAGE_UNSPECIFIED: Do not use this default value.
      UNIMPLEMENTED: The feature is not yet implemented. Users can not use it.
      PRELAUNCH: Prelaunch features are hidden from users and are only visible
        internally.
      EARLY_ACCESS: Early Access features are limited to a closed group of
        testers. To use these features, you must sign up in advance and sign a
        Trusted Tester agreement (which includes confidentiality provisions).
        These features may be unstable, changed in backward-incompatible ways,
        and are not guaranteed to be released.
      ALPHA: Alpha is a limited availability test for releases before they are
        cleared for widespread use. By Alpha, all significant design issues
        are resolved and we are in the process of verifying functionality.
        Alpha customers need to apply for access, agree to applicable terms,
        and have their projects allowlisted. Alpha releases don't have to be
        feature complete, no SLAs are provided, and there are no technical
        support obligations, but they will be far enough along that customers
        can actually use them in test environments or for limited-use tests --
        just like they would in normal production cases.
      BETA: Beta is the point at which we are ready to open a release for any
        customer to use. There are no SLA or technical support obligations in
        a Beta release. Products will be complete from a feature perspective,
        but may have some open outstanding issues. Beta releases are suitable
        for limited production use cases.
      GA: GA features are open to all developers and are considered stable and
        fully qualified for production use.
      DEPRECATED: Deprecated features are scheduled to be shut down and
        removed. For more information, see the "Deprecation Policy" section of
        our [Terms of Service](https://cloud.google.com/terms/) and the
        [Google Cloud Platform Subject to the Deprecation
        Policy](https://cloud.google.com/terms/deprecation) documentation.
    """
        LAUNCH_STAGE_UNSPECIFIED = 0
        UNIMPLEMENTED = 1
        PRELAUNCH = 2
        EARLY_ACCESS = 3
        ALPHA = 4
        BETA = 5
        GA = 6
        DEPRECATED = 7

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Output only. Unstructured key value map that may be set by external
    tools to store and arbitrary metadata. They are not queryable and should
    be preserved when modifying objects.

    Messages:
      AdditionalProperty: An additional property for a AnnotationsValue
        object.

    Fields:
      additionalProperties: Additional properties of type AnnotationsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AnnotationsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Output only. Unstructured key value map that can be used to organize
    and categorize objects. User-provided labels are shared with Google's
    billing system, so they can be used to filter, or break down billing
    charges by team, component, environment, state, etc. For more information,
    visit https://cloud.google.com/resource-manager/docs/creating-managing-
    labels or https://cloud.google.com/run/docs/configuring/labels

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    annotations = _messages.MessageField('AnnotationsValue', 1)
    cancelledCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    completionTime = _messages.StringField(3)
    conditions = _messages.MessageField('GoogleCloudRunV2Condition', 4, repeated=True)
    createTime = _messages.StringField(5)
    deleteTime = _messages.StringField(6)
    etag = _messages.StringField(7)
    expireTime = _messages.StringField(8)
    failedCount = _messages.IntegerField(9, variant=_messages.Variant.INT32)
    generation = _messages.IntegerField(10)
    job = _messages.StringField(11)
    labels = _messages.MessageField('LabelsValue', 12)
    launchStage = _messages.EnumField('LaunchStageValueValuesEnum', 13)
    logUri = _messages.StringField(14)
    name = _messages.StringField(15)
    observedGeneration = _messages.IntegerField(16)
    parallelism = _messages.IntegerField(17, variant=_messages.Variant.INT32)
    reconciling = _messages.BooleanField(18)
    retriedCount = _messages.IntegerField(19, variant=_messages.Variant.INT32)
    runningCount = _messages.IntegerField(20, variant=_messages.Variant.INT32)
    satisfiesPzs = _messages.BooleanField(21)
    startTime = _messages.StringField(22)
    succeededCount = _messages.IntegerField(23, variant=_messages.Variant.INT32)
    taskCount = _messages.IntegerField(24, variant=_messages.Variant.INT32)
    template = _messages.MessageField('GoogleCloudRunV2TaskTemplate', 25)
    uid = _messages.StringField(26)
    updateTime = _messages.StringField(27)