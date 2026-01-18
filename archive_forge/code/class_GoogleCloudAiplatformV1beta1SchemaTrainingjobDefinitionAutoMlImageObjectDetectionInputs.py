from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlImageObjectDetectionInputs(_messages.Message):
    """A GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlImageObje
  ctDetectionInputs object.

  Enums:
    ModelTypeValueValuesEnum:

  Fields:
    budgetMilliNodeHours: The training budget of creating this model,
      expressed in milli node hours i.e. 1,000 value in this field means 1
      node hour. The actual metadata.costMilliNodeHours will be equal or less
      than this value. If further model training ceases to provide any
      improvements, it will stop without using the full budget and the
      metadata.successfulStopReason will be `model-converged`. Note, node_hour
      = actual_hour * number_of_nodes_involved. For modelType
      `cloud`(default), the budget must be between 20,000 and 900,000 milli
      node hours, inclusive. The default value is 216,000 which represents one
      day in wall time, considering 9 nodes are used. For model types `mobile-
      tf-low-latency-1`, `mobile-tf-versatile-1`, `mobile-tf-high-accuracy-1`
      the training budget must be between 1,000 and 100,000 milli node hours,
      inclusive. The default value is 24,000 which represents one day in wall
      time on a single node that is used.
    disableEarlyStopping: Use the entire training budget. This disables the
      early stopping feature. When false the early stopping feature is
      enabled, which means that AutoML Image Object Detection might stop
      training before the entire training budget has been used.
    modelType: A ModelTypeValueValuesEnum attribute.
    tunableParameter: Trainer type for Vision TrainRequest.
    uptrainBaseModelId: The ID of `base` model for upTraining. If it is
      specified, the new model will be upTrained based on the `base` model for
      upTraining. Otherwise, the new model will be trained from scratch. The
      `base` model for upTraining must be in the same Project and Location as
      the new Model to train, and have the same modelType.
  """

    class ModelTypeValueValuesEnum(_messages.Enum):
        """ModelTypeValueValuesEnum enum type.

    Values:
      MODEL_TYPE_UNSPECIFIED: Should not be set.
      CLOUD_HIGH_ACCURACY_1: A model best tailored to be used within Google
        Cloud, and which cannot be exported. Expected to have a higher
        latency, but should also have a higher prediction quality than other
        cloud models.
      CLOUD_LOW_LATENCY_1: A model best tailored to be used within Google
        Cloud, and which cannot be exported. Expected to have a low latency,
        but may have lower prediction quality than other cloud models.
      CLOUD_1: A model best tailored to be used within Google Cloud, and which
        cannot be exported. Compared to the CLOUD_HIGH_ACCURACY_1 and
        CLOUD_LOW_LATENCY_1 models above, it is expected to have higher
        prediction quality and lower latency.
      MOBILE_TF_LOW_LATENCY_1: A model that, in addition to being available
        within Google Cloud can also be exported (see
        ModelService.ExportModel) and used on a mobile or edge device with
        TensorFlow afterwards. Expected to have low latency, but may have
        lower prediction quality than other mobile models.
      MOBILE_TF_VERSATILE_1: A model that, in addition to being available
        within Google Cloud can also be exported (see
        ModelService.ExportModel) and used on a mobile or edge device with
        TensorFlow afterwards.
      MOBILE_TF_HIGH_ACCURACY_1: A model that, in addition to being available
        within Google Cloud, can also be exported (see
        ModelService.ExportModel) and used on a mobile or edge device with
        TensorFlow afterwards. Expected to have a higher latency, but should
        also have a higher prediction quality than other mobile models.
      CLOUD_STREAMING_1: A model best tailored to be used within Google Cloud,
        and which cannot be exported. Expected to best support predictions in
        streaming with lower latency and lower prediction quality than other
        cloud models.
      SPINENET: SpineNet for Model Garden training with customizable
        hyperparameters. Best tailored to be used within Google Cloud, and
        cannot be exported externally.
      YOLO: YOLO for Model Garden training with customizable hyperparameters.
        Best tailored to be used within Google Cloud, and cannot be exported
        externally.
    """
        MODEL_TYPE_UNSPECIFIED = 0
        CLOUD_HIGH_ACCURACY_1 = 1
        CLOUD_LOW_LATENCY_1 = 2
        CLOUD_1 = 3
        MOBILE_TF_LOW_LATENCY_1 = 4
        MOBILE_TF_VERSATILE_1 = 5
        MOBILE_TF_HIGH_ACCURACY_1 = 6
        CLOUD_STREAMING_1 = 7
        SPINENET = 8
        YOLO = 9
    budgetMilliNodeHours = _messages.IntegerField(1)
    disableEarlyStopping = _messages.BooleanField(2)
    modelType = _messages.EnumField('ModelTypeValueValuesEnum', 3)
    tunableParameter = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutomlImageTrainingTunableParameter', 4)
    uptrainBaseModelId = _messages.StringField(5)