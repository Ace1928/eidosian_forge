from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2ExecutionTemplate(_messages.Message):
    """ExecutionTemplate describes the data an execution should have when
  created from a template.

  Messages:
    AnnotationsValue: Unstructured key value map that may be set by external
      tools to store and arbitrary metadata. They are not queryable and should
      be preserved when modifying objects. Cloud Run API v2 does not support
      annotations with `run.googleapis.com`, `cloud.googleapis.com`,
      `serving.knative.dev`, or `autoscaling.knative.dev` namespaces, and they
      will be rejected. All system annotations in v1 now have a corresponding
      field in v2 ExecutionTemplate. This field follows Kubernetes
      annotations' namespacing, limits, and rules.
    LabelsValue: Unstructured key value map that can be used to organize and
      categorize objects. User-provided labels are shared with Google's
      billing system, so they can be used to filter, or break down billing
      charges by team, component, environment, state, etc. For more
      information, visit https://cloud.google.com/resource-
      manager/docs/creating-managing-labels or
      https://cloud.google.com/run/docs/configuring/labels. Cloud Run API v2
      does not support labels with `run.googleapis.com`,
      `cloud.googleapis.com`, `serving.knative.dev`, or
      `autoscaling.knative.dev` namespaces, and they will be rejected. All
      system labels in v1 now have a corresponding field in v2
      ExecutionTemplate.

  Fields:
    annotations: Unstructured key value map that may be set by external tools
      to store and arbitrary metadata. They are not queryable and should be
      preserved when modifying objects. Cloud Run API v2 does not support
      annotations with `run.googleapis.com`, `cloud.googleapis.com`,
      `serving.knative.dev`, or `autoscaling.knative.dev` namespaces, and they
      will be rejected. All system annotations in v1 now have a corresponding
      field in v2 ExecutionTemplate. This field follows Kubernetes
      annotations' namespacing, limits, and rules.
    labels: Unstructured key value map that can be used to organize and
      categorize objects. User-provided labels are shared with Google's
      billing system, so they can be used to filter, or break down billing
      charges by team, component, environment, state, etc. For more
      information, visit https://cloud.google.com/resource-
      manager/docs/creating-managing-labels or
      https://cloud.google.com/run/docs/configuring/labels. Cloud Run API v2
      does not support labels with `run.googleapis.com`,
      `cloud.googleapis.com`, `serving.knative.dev`, or
      `autoscaling.knative.dev` namespaces, and they will be rejected. All
      system labels in v1 now have a corresponding field in v2
      ExecutionTemplate.
    parallelism: Specifies the maximum desired number of tasks the execution
      should run at given time. Must be <= task_count. When the job is run, if
      this field is 0 or unset, the maximum possible value will be used for
      that execution. The actual number of tasks running in steady state will
      be less than this number when there are fewer tasks waiting to be
      completed remaining, i.e. when the work left to do is less than max
      parallelism.
    taskCount: Specifies the desired number of tasks the execution should run.
      Setting to 1 means that parallelism is limited to 1 and the success of
      that task signals the success of the execution. Defaults to 1.
    template: Required. Describes the task(s) that will be created when
      executing an execution.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Unstructured key value map that may be set by external tools to store
    and arbitrary metadata. They are not queryable and should be preserved
    when modifying objects. Cloud Run API v2 does not support annotations with
    `run.googleapis.com`, `cloud.googleapis.com`, `serving.knative.dev`, or
    `autoscaling.knative.dev` namespaces, and they will be rejected. All
    system annotations in v1 now have a corresponding field in v2
    ExecutionTemplate. This field follows Kubernetes annotations' namespacing,
    limits, and rules.

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
        """Unstructured key value map that can be used to organize and categorize
    objects. User-provided labels are shared with Google's billing system, so
    they can be used to filter, or break down billing charges by team,
    component, environment, state, etc. For more information, visit
    https://cloud.google.com/resource-manager/docs/creating-managing-labels or
    https://cloud.google.com/run/docs/configuring/labels. Cloud Run API v2
    does not support labels with `run.googleapis.com`, `cloud.googleapis.com`,
    `serving.knative.dev`, or `autoscaling.knative.dev` namespaces, and they
    will be rejected. All system labels in v1 now have a corresponding field
    in v2 ExecutionTemplate.

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
    labels = _messages.MessageField('LabelsValue', 2)
    parallelism = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    taskCount = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    template = _messages.MessageField('GoogleCloudRunV2TaskTemplate', 5)