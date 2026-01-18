from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ModelMonitoringObjectiveConfigTrainingDataset(_messages.Message):
    """Training Dataset information.

  Fields:
    bigquerySource: The BigQuery table of the unmanaged Dataset used to train
      this Model.
    dataFormat: Data format of the dataset, only applicable if the input is
      from Google Cloud Storage. The possible formats are: "tf-record" The
      source file is a TFRecord file. "csv" The source file is a CSV file.
      "jsonl" The source file is a JSONL file.
    dataset: The resource name of the Dataset used to train this Model.
    gcsSource: The Google Cloud Storage uri of the unmanaged Dataset used to
      train this Model.
    loggingSamplingStrategy: Strategy to sample data from Training Dataset. If
      not set, we process the whole dataset.
    targetField: The target field name the model is to predict. This field
      will be excluded when doing Predict and (or) Explain for the training
      data.
  """
    bigquerySource = _messages.MessageField('GoogleCloudAiplatformV1BigQuerySource', 1)
    dataFormat = _messages.StringField(2)
    dataset = _messages.StringField(3)
    gcsSource = _messages.MessageField('GoogleCloudAiplatformV1GcsSource', 4)
    loggingSamplingStrategy = _messages.MessageField('GoogleCloudAiplatformV1SamplingStrategy', 5)
    targetField = _messages.StringField(6)