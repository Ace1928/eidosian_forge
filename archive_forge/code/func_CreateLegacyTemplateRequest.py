from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def CreateLegacyTemplateRequest(self, args):
    """Create a Legacy Template request for the Pipeline workload.

    Args:
      args: Any, list of args needed to create a Pipeline.

    Returns:
      Legacy Template request.
    """
    location = args.region
    project_id = properties.VALUES.core.project.Get(required=True)
    params_list = self.ConvertDictArguments(args.parameters, self.messages.GoogleCloudDatapipelinesV1LaunchTemplateParameters.ParametersValue)
    transform_mapping_list = self.ConvertDictArguments(args.transform_name_mappings, self.messages.GoogleCloudDatapipelinesV1LaunchTemplateParameters.TransformNameMappingValue)
    transform_name_mappings = None
    if transform_mapping_list:
        transform_name_mappings = self.messages.GoogleCloudDatapipelinesV1LaunchTemplateParameters.TransformNameMappingValue(additionalProperties=transform_mapping_list)
    ip_private = self.messages.GoogleCloudDatapipelinesV1RuntimeEnvironment.IpConfigurationValueValuesEnum.WORKER_IP_PRIVATE
    ip_configuration = ip_private if args.disable_public_ips else None
    user_labels_list = self.ConvertDictArguments(args.additional_user_labels, self.messages.GoogleCloudDatapipelinesV1RuntimeEnvironment.AdditionalUserLabelsValue)
    additional_user_labels = None
    if user_labels_list:
        additional_user_labels = self.messages.GoogleCloudDatapipelinesV1RuntimeEnvironment.AdditionalUserLabelsValue(additionalProperties=user_labels_list)
    launch_parameter = self.messages.GoogleCloudDatapipelinesV1LaunchTemplateParameters(environment=self.messages.GoogleCloudDatapipelinesV1RuntimeEnvironment(serviceAccountEmail=args.dataflow_service_account_email, maxWorkers=args.max_workers, numWorkers=args.num_workers, network=args.network, subnetwork=args.subnetwork, machineType=args.worker_machine_type, tempLocation=args.temp_location, kmsKeyName=args.dataflow_kms_key, ipConfiguration=ip_configuration, workerRegion=args.worker_region, workerZone=args.worker_zone, enableStreamingEngine=args.enable_streaming_engine, additionalExperiments=args.additional_experiments if args.additional_experiments else [], additionalUserLabels=additional_user_labels), update=args.update, parameters=self.messages.GoogleCloudDatapipelinesV1LaunchTemplateParameters.ParametersValue(additionalProperties=params_list) if params_list else None, transformNameMapping=transform_name_mappings)
    return self.messages.GoogleCloudDatapipelinesV1LaunchTemplateRequest(gcsPath=args.template_file_gcs_location, location=location, projectId=project_id, launchParameters=launch_parameter)