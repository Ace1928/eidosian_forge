from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import extra_types
from googlecloudsdk.api_lib.ai import operations
from googlecloudsdk.api_lib.ai.models import client
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.command_lib.ai import models_util
from googlecloudsdk.command_lib.ai import operations_util
from googlecloudsdk.command_lib.ai import region_util
from googlecloudsdk.core import yaml
def _BuildExplanationSpec(self, args):
    parameters = None
    method = args.explanation_method
    if not method:
        return None
    if method.lower() == 'integrated-gradients':
        parameters = self.messages.GoogleCloudAiplatformV1beta1ExplanationParameters(integratedGradientsAttribution=self.messages.GoogleCloudAiplatformV1beta1IntegratedGradientsAttribution(stepCount=args.explanation_step_count, smoothGradConfig=self._BuildSmoothGradConfig(args)))
    elif method.lower() == 'xrai':
        parameters = self.messages.GoogleCloudAiplatformV1beta1ExplanationParameters(xraiAttribution=self.messages.GoogleCloudAiplatformV1beta1XraiAttribution(stepCount=args.explanation_step_count, smoothGradConfig=self._BuildSmoothGradConfig(args)))
    elif method.lower() == 'sampled-shapley':
        parameters = self.messages.GoogleCloudAiplatformV1beta1ExplanationParameters(sampledShapleyAttribution=self.messages.GoogleCloudAiplatformV1beta1SampledShapleyAttribution(pathCount=args.explanation_path_count))
    elif method.lower() == 'examples':
        if args.explanation_nearest_neighbor_search_config_file:
            parameters = self.messages.GoogleCloudAiplatformV1beta1ExplanationParameters(examples=self.messages.GoogleCloudAiplatformV1beta1Examples(gcsSource=self.messages.GoogleCloudAiplatformV1beta1GcsSource(uris=args.uris), neighborCount=args.explanation_neighbor_count, nearestNeighborSearchConfig=self._ReadIndexMetadata(args.explanation_nearest_neighbor_search_config_file)))
        else:
            parameters = self.messages.GoogleCloudAiplatformV1beta1ExplanationParameters(examples=self.messages.GoogleCloudAiplatformV1beta1Examples(gcsSource=self.messages.GoogleCloudAiplatformV1beta1GcsSource(uris=args.uris), neighborCount=args.explanation_neighbor_count, presets=self.messages.GoogleCloudAiplatformV1beta1Presets(modality=self.messages.GoogleCloudAiplatformV1beta1Presets.ModalityValueValuesEnum(args.explanation_modality), query=self.messages.GoogleCloudAiplatformV1beta1Presets.QueryValueValuesEnum(args.explanation_query))))
    else:
        raise gcloud_exceptions.BadArgumentException('--explanation-method', 'Explanation method must be one of `integrated-gradients`, `xrai`, `sampled-shapley` and `examples`.')
    return self.messages.GoogleCloudAiplatformV1beta1ExplanationSpec(metadata=self._ReadExplanationMetadata(args.explanation_metadata_file), parameters=parameters)