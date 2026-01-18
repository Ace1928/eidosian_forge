from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
import textwrap
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai import region_util
from googlecloudsdk.command_lib.iam import iam_util as core_iam_util
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetMonitoringConfigFromFile():
    return base.Argument('--monitoring-config-from-file', help="\nPath to the model monitoring objective config file. This file should be a YAML\ndocument containing a `ModelDeploymentMonitoringJob`(https://cloud.google.com/vertex-ai/docs/reference/rest/v1beta1/projects.locations.modelDeploymentMonitoringJobs#ModelDeploymentMonitoringJob),\nbut only the ModelDeploymentMonitoringObjectiveConfig needs to be configured.\n\nNote: Only one of --monitoring-config-from-file and other objective config set,\nlike --feature-thresholds, --feature-attribution-thresholds needs to be set.\n\nExample(YAML):\n\n  modelDeploymentMonitoringObjectiveConfigs:\n  - deployedModelId: '5251549009234886656'\n    objectiveConfig:\n      trainingDataset:\n        dataFormat: csv\n        gcsSource:\n          uris:\n          - gs://fake-bucket/training_data.csv\n        targetField: price\n      trainingPredictionSkewDetectionConfig:\n        skewThresholds:\n          feat1:\n            value: 0.9\n          feat2:\n            value: 0.8\n  - deployedModelId: '2945706000021192704'\n    objectiveConfig:\n      predictionDriftDetectionConfig:\n        driftThresholds:\n          feat1:\n            value: 0.3\n          feat2:\n            value: 0.4\n")