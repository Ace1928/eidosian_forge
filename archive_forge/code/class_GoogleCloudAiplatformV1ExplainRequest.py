from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ExplainRequest(_messages.Message):
    """Request message for PredictionService.Explain.

  Fields:
    deployedModelId: If specified, this ExplainRequest will be served by the
      chosen DeployedModel, overriding Endpoint.traffic_split.
    explanationSpecOverride: If specified, overrides the explanation_spec of
      the DeployedModel. Can be used for explaining prediction results with
      different configurations, such as: - Explaining top-5 predictions
      results as opposed to top-1; - Increasing path count or step count of
      the attribution methods to reduce approximate errors; - Using different
      baselines for explaining the prediction results.
    instances: Required. The instances that are the input to the explanation
      call. A DeployedModel may have an upper limit on the number of instances
      it supports per request, and when it is exceeded the explanation call
      errors in case of AutoML Models, or, in case of customer created Models,
      the behaviour is as documented by that Model. The schema of any single
      instance may be specified via Endpoint's DeployedModels' Model's
      PredictSchemata's instance_schema_uri.
    parameters: The parameters that govern the prediction. The schema of the
      parameters may be specified via Endpoint's DeployedModels' Model's
      PredictSchemata's parameters_schema_uri.
  """
    deployedModelId = _messages.StringField(1)
    explanationSpecOverride = _messages.MessageField('GoogleCloudAiplatformV1ExplanationSpecOverride', 2)
    instances = _messages.MessageField('extra_types.JsonValue', 3, repeated=True)
    parameters = _messages.MessageField('extra_types.JsonValue', 4)