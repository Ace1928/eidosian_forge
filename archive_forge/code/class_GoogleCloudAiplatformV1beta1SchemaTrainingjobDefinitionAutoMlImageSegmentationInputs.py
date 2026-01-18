from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlImageSegmentationInputs(_messages.Message):
    """A GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlImageSegm
  entationInputs object.

  Enums:
    ModelTypeValueValuesEnum:

  Fields:
    baseModelId: The ID of the `base` model. If it is specified, the new model
      will be trained based on the `base` model. Otherwise, the new model will
      be trained from scratch. The `base` model must be in the same Project
      and Location as the new Model to train, and have the same modelType.
    budgetMilliNodeHours: The training budget of creating this model,
      expressed in milli node hours i.e. 1,000 value in this field means 1
      node hour. The actual metadata.costMilliNodeHours will be equal or less
      than this value. If further model training ceases to provide any
      improvements, it will stop without using the full budget and the
      metadata.successfulStopReason will be `model-converged`. Note, node_hour
      = actual_hour * number_of_nodes_involved. Or actual_wall_clock_hours =
      train_budget_milli_node_hours / (number_of_nodes_involved * 1000) For
      modelType `cloud-high-accuracy-1`(default), the budget must be between
      20,000 and 2,000,000 milli node hours, inclusive. The default value is
      192,000 which represents one day in wall time (1000 milli * 24 hours * 8
      nodes).
    modelType: A ModelTypeValueValuesEnum attribute.
  """

    class ModelTypeValueValuesEnum(_messages.Enum):
        """ModelTypeValueValuesEnum enum type.

    Values:
      MODEL_TYPE_UNSPECIFIED: Should not be set.
      CLOUD_HIGH_ACCURACY_1: A model to be used via prediction calls to uCAIP
        API. Expected to have a higher latency, but should also have a higher
        prediction quality than other models.
      CLOUD_LOW_ACCURACY_1: A model to be used via prediction calls to uCAIP
        API. Expected to have a lower latency but relatively lower prediction
        quality.
      MOBILE_TF_LOW_LATENCY_1: A model that, in addition to being available
        within Google Cloud, can also be exported (see
        ModelService.ExportModel) as TensorFlow model and used on a mobile or
        edge device afterwards. Expected to have low latency, but may have
        lower prediction quality than other mobile models.
    """
        MODEL_TYPE_UNSPECIFIED = 0
        CLOUD_HIGH_ACCURACY_1 = 1
        CLOUD_LOW_ACCURACY_1 = 2
        MOBILE_TF_LOW_LATENCY_1 = 3
    baseModelId = _messages.StringField(1)
    budgetMilliNodeHours = _messages.IntegerField(2)
    modelType = _messages.EnumField('ModelTypeValueValuesEnum', 3)