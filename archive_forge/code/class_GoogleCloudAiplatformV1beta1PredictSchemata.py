from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1PredictSchemata(_messages.Message):
    """Contains the schemata used in Model's predictions and explanations via
  PredictionService.Predict, PredictionService.Explain and BatchPredictionJob.

  Fields:
    instanceSchemaUri: Immutable. Points to a YAML file stored on Google Cloud
      Storage describing the format of a single instance, which are used in
      PredictRequest.instances, ExplainRequest.instances and
      BatchPredictionJob.input_config. The schema is defined as an OpenAPI
      3.0.2 [Schema Object](https://github.com/OAI/OpenAPI-
      Specification/blob/main/versions/3.0.2.md#schemaObject). AutoML Models
      always have this field populated by Vertex AI. Note: The URI given on
      output will be immutable and probably different, including the URI
      scheme, than the one given on input. The output URI will point to a
      location where the user only has a read access.
    parametersSchemaUri: Immutable. Points to a YAML file stored on Google
      Cloud Storage describing the parameters of prediction and explanation
      via PredictRequest.parameters, ExplainRequest.parameters and
      BatchPredictionJob.model_parameters. The schema is defined as an OpenAPI
      3.0.2 [Schema Object](https://github.com/OAI/OpenAPI-
      Specification/blob/main/versions/3.0.2.md#schemaObject). AutoML Models
      always have this field populated by Vertex AI, if no parameters are
      supported, then it is set to an empty string. Note: The URI given on
      output will be immutable and probably different, including the URI
      scheme, than the one given on input. The output URI will point to a
      location where the user only has a read access.
    predictionSchemaUri: Immutable. Points to a YAML file stored on Google
      Cloud Storage describing the format of a single prediction produced by
      this Model, which are returned via PredictResponse.predictions,
      ExplainResponse.explanations, and BatchPredictionJob.output_config. The
      schema is defined as an OpenAPI 3.0.2 [Schema
      Object](https://github.com/OAI/OpenAPI-
      Specification/blob/main/versions/3.0.2.md#schemaObject). AutoML Models
      always have this field populated by Vertex AI. Note: The URI given on
      output will be immutable and probably different, including the URI
      scheme, than the one given on input. The output URI will point to a
      location where the user only has a read access.
  """
    instanceSchemaUri = _messages.StringField(1)
    parametersSchemaUri = _messages.StringField(2)
    predictionSchemaUri = _messages.StringField(3)