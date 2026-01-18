from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from apitools.base.py import encoding
from apitools.base.py import extra_types
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.ai import util as api_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai import model_monitoring_jobs_util
from googlecloudsdk.command_lib.ai import validation as common_validation
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
import six
def _ConstructObjectiveConfigForCreate(self, location_ref, endpoint_name, feature_thresholds, feature_attribution_thresholds, dataset, bigquery_uri, data_format, gcs_uris, target_field, training_sampling_rate):
    """Construct monitoring objective config.

    Apply the feature thresholds for skew or drift detection to all the deployed
    models under the endpoint.
    Args:
      location_ref: Location reference.
      endpoint_name: Endpoint resource name.
      feature_thresholds: Dict or None, key: feature_name, value: thresholds.
      feature_attribution_thresholds: Dict or None, key: feature_name, value:
        attribution score thresholds.
      dataset: Vertex AI Dataset Id.
      bigquery_uri: The BigQuery table of the unmanaged Dataset used to train
        this Model.
      data_format: Google Cloud Storage format, supported format: csv,
        tf-record.
      gcs_uris: The Google Cloud Storage uri of the unmanaged Dataset used to
        train this Model.
      target_field: The target field name the model is to predict.
      training_sampling_rate: Training Dataset sampling rate.

    Returns:
      A list of model monitoring objective config.
    """
    objective_config_template = api_util.GetMessage('ModelDeploymentMonitoringObjectiveConfig', self._version)()
    if feature_thresholds or feature_attribution_thresholds:
        if dataset or bigquery_uri or gcs_uris or data_format:
            training_dataset = api_util.GetMessage('ModelMonitoringObjectiveConfigTrainingDataset', self._version)()
            if target_field is None:
                raise errors.ArgumentError("Target field must be provided if you'd like to do training-prediction skew detection.")
            training_dataset.targetField = target_field
            training_dataset.loggingSamplingStrategy = api_util.GetMessage('SamplingStrategy', self._version)(randomSampleConfig=api_util.GetMessage('SamplingStrategyRandomSampleConfig', self._version)(sampleRate=training_sampling_rate))
            if dataset:
                training_dataset.dataset = _ParseDataset(dataset, location_ref).RelativeName()
            elif bigquery_uri:
                training_dataset.bigquerySource = api_util.GetMessage('BigQuerySource', self._version)(inputUri=bigquery_uri)
            elif gcs_uris or data_format:
                if gcs_uris is None:
                    raise errors.ArgumentError('Data format is defined but no Google Cloud Storage uris are provided. Please use --gcs-uris to provide training datasets.')
                if data_format is None:
                    raise errors.ArgumentError('No Data format is defined for Google Cloud Storage training dataset. Please use --data-format to define the Data format.')
                training_dataset.dataFormat = data_format
                training_dataset.gcsSource = api_util.GetMessage('GcsSource', self._version)(uris=gcs_uris)
            training_prediction_skew_detection = self._ConstructSkewThresholds(feature_thresholds, feature_attribution_thresholds)
            objective_config_template.objectiveConfig = api_util.GetMessage('ModelMonitoringObjectiveConfig', self._version)(trainingDataset=training_dataset, trainingPredictionSkewDetectionConfig=training_prediction_skew_detection)
        else:
            prediction_drift_detection = self._ConstructDriftThresholds(feature_thresholds, feature_attribution_thresholds)
            objective_config_template.objectiveConfig = api_util.GetMessage('ModelMonitoringObjectiveConfig', self._version)(predictionDriftDetectionConfig=prediction_drift_detection)
        if feature_attribution_thresholds:
            objective_config_template.objectiveConfig.explanationConfig = api_util.GetMessage('ModelMonitoringObjectiveConfigExplanationConfig', self._version)(enableFeatureAttributes=True)
    get_endpoint_req = self.messages.AiplatformProjectsLocationsEndpointsGetRequest(name=endpoint_name)
    endpoint = self.client.projects_locations_endpoints.Get(get_endpoint_req)
    objective_configs = []
    for deployed_model in endpoint.deployedModels:
        objective_config = copy.deepcopy(objective_config_template)
        objective_config.deployedModelId = deployed_model.id
        objective_configs.append(objective_config)
    return objective_configs