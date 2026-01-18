from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1TensorboardRun(_messages.Message):
    """TensorboardRun maps to a specific execution of a training job with a
  given set of hyperparameter values, model definition, dataset, etc

  Messages:
    LabelsValue: The labels with user-defined metadata to organize your
      TensorboardRuns. This field will be used to filter and visualize Runs in
      the Tensorboard UI. For example, a Vertex AI training job can set a
      label aiplatform.googleapis.com/training_job_id=xxxxx to all the runs
      created within that job. An end user can set a label experiment_id=xxxxx
      for all the runs produced in a Jupyter notebook. These runs can be
      grouped by a label value and visualized together in the Tensorboard UI.
      Label keys and values can be no longer than 64 characters (Unicode
      codepoints), can only contain lowercase letters, numeric characters,
      underscores and dashes. International characters are allowed. No more
      than 64 user labels can be associated with one TensorboardRun (System
      labels are excluded). See https://goo.gl/xmQnxf for more information and
      examples of labels. System reserved label keys are prefixed with
      "aiplatform.googleapis.com/" and are immutable.

  Fields:
    createTime: Output only. Timestamp when this TensorboardRun was created.
    description: Description of this TensorboardRun.
    displayName: Required. User provided name of this TensorboardRun. This
      value must be unique among all TensorboardRuns belonging to the same
      parent TensorboardExperiment.
    etag: Used to perform a consistent read-modify-write updates. If not set,
      a blind "overwrite" update happens.
    labels: The labels with user-defined metadata to organize your
      TensorboardRuns. This field will be used to filter and visualize Runs in
      the Tensorboard UI. For example, a Vertex AI training job can set a
      label aiplatform.googleapis.com/training_job_id=xxxxx to all the runs
      created within that job. An end user can set a label experiment_id=xxxxx
      for all the runs produced in a Jupyter notebook. These runs can be
      grouped by a label value and visualized together in the Tensorboard UI.
      Label keys and values can be no longer than 64 characters (Unicode
      codepoints), can only contain lowercase letters, numeric characters,
      underscores and dashes. International characters are allowed. No more
      than 64 user labels can be associated with one TensorboardRun (System
      labels are excluded). See https://goo.gl/xmQnxf for more information and
      examples of labels. System reserved label keys are prefixed with
      "aiplatform.googleapis.com/" and are immutable.
    name: Output only. Name of the TensorboardRun. Format: `projects/{project}
      /locations/{location}/tensorboards/{tensorboard}/experiments/{experiment
      }/runs/{run}`
    updateTime: Output only. Timestamp when this TensorboardRun was last
      updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The labels with user-defined metadata to organize your
    TensorboardRuns. This field will be used to filter and visualize Runs in
    the Tensorboard UI. For example, a Vertex AI training job can set a label
    aiplatform.googleapis.com/training_job_id=xxxxx to all the runs created
    within that job. An end user can set a label experiment_id=xxxxx for all
    the runs produced in a Jupyter notebook. These runs can be grouped by a
    label value and visualized together in the Tensorboard UI. Label keys and
    values can be no longer than 64 characters (Unicode codepoints), can only
    contain lowercase letters, numeric characters, underscores and dashes.
    International characters are allowed. No more than 64 user labels can be
    associated with one TensorboardRun (System labels are excluded). See
    https://goo.gl/xmQnxf for more information and examples of labels. System
    reserved label keys are prefixed with "aiplatform.googleapis.com/" and are
    immutable.

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
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    etag = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    name = _messages.StringField(6)
    updateTime = _messages.StringField(7)