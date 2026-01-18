from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1TensorboardExperiment(_messages.Message):
    """A TensorboardExperiment is a group of TensorboardRuns, that are
  typically the results of a training job run, in a Tensorboard.

  Messages:
    LabelsValue: The labels with user-defined metadata to organize your
      TensorboardExperiment. Label keys and values cannot be longer than 64
      characters (Unicode codepoints), can only contain lowercase letters,
      numeric characters, underscores and dashes. International characters are
      allowed. No more than 64 user labels can be associated with one Dataset
      (System labels are excluded). See https://goo.gl/xmQnxf for more
      information and examples of labels. System reserved label keys are
      prefixed with `aiplatform.googleapis.com/` and are immutable. The
      following system labels exist for each Dataset: *
      `aiplatform.googleapis.com/dataset_metadata_schema`: output only. Its
      value is the metadata_schema's title.

  Fields:
    createTime: Output only. Timestamp when this TensorboardExperiment was
      created.
    description: Description of this TensorboardExperiment.
    displayName: User provided name of this TensorboardExperiment.
    etag: Used to perform consistent read-modify-write updates. If not set, a
      blind "overwrite" update happens.
    labels: The labels with user-defined metadata to organize your
      TensorboardExperiment. Label keys and values cannot be longer than 64
      characters (Unicode codepoints), can only contain lowercase letters,
      numeric characters, underscores and dashes. International characters are
      allowed. No more than 64 user labels can be associated with one Dataset
      (System labels are excluded). See https://goo.gl/xmQnxf for more
      information and examples of labels. System reserved label keys are
      prefixed with `aiplatform.googleapis.com/` and are immutable. The
      following system labels exist for each Dataset: *
      `aiplatform.googleapis.com/dataset_metadata_schema`: output only. Its
      value is the metadata_schema's title.
    name: Output only. Name of the TensorboardExperiment. Format: `projects/{p
      roject}/locations/{location}/tensorboards/{tensorboard}/experiments/{exp
      eriment}`
    source: Immutable. Source of the TensorboardExperiment. Example: a custom
      training job.
    updateTime: Output only. Timestamp when this TensorboardExperiment was
      last updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The labels with user-defined metadata to organize your
    TensorboardExperiment. Label keys and values cannot be longer than 64
    characters (Unicode codepoints), can only contain lowercase letters,
    numeric characters, underscores and dashes. International characters are
    allowed. No more than 64 user labels can be associated with one Dataset
    (System labels are excluded). See https://goo.gl/xmQnxf for more
    information and examples of labels. System reserved label keys are
    prefixed with `aiplatform.googleapis.com/` and are immutable. The
    following system labels exist for each Dataset: *
    `aiplatform.googleapis.com/dataset_metadata_schema`: output only. Its
    value is the metadata_schema's title.

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
    source = _messages.StringField(7)
    updateTime = _messages.StringField(8)