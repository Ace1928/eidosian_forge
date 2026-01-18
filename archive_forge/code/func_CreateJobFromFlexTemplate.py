from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import shutil
import textwrap
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.command_lib.builds import submit_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
@staticmethod
def CreateJobFromFlexTemplate(template_args=None):
    """Calls the create job from flex template APIs.

    Args:
      template_args: Arguments for create template.

    Returns:
      (Job)
    """
    params_list = Templates.__ConvertDictArguments(template_args.parameters, Templates.FLEX_TEMPLATE_PARAMETERS_VALUE)
    transform_mapping_list = Templates.__ConvertDictArguments(template_args.transform_name_mappings, Templates.FLEX_TEMPLATE_TRANSFORM_NAME_MAPPING_VALUE)
    transform_mappings = None
    streaming_update = None
    if template_args.streaming_update:
        streaming_update = template_args.streaming_update
        if transform_mapping_list:
            transform_mappings = Templates.FLEX_TEMPLATE_TRANSFORM_NAME_MAPPING_VALUE(additionalProperties=transform_mapping_list)
    user_labels_list = Templates.__ConvertDictArguments(template_args.additional_user_labels, Templates.FLEX_TEMPLATE_USER_LABELS_VALUE)
    region_id = template_args.region_id or DATAFLOW_API_DEFAULT_REGION
    ip_private = Templates.IP_CONFIGURATION_ENUM_VALUE.WORKER_IP_PRIVATE
    ip_configuration = ip_private if template_args.disable_public_ips else None
    flexrs_goal = None
    if template_args.flexrs_goal:
        if template_args.flexrs_goal == 'SPEED_OPTIMIZED':
            flexrs_goal = Templates.FLEXRS_GOAL_ENUM_VALUE.FLEXRS_SPEED_OPTIMIZED
        elif template_args.flexrs_goal == 'COST_OPTIMIZED':
            flexrs_goal = Templates.FLEXRS_GOAL_ENUM_VALUE.FLEXRS_COST_OPTIMIZED
    body = Templates.LAUNCH_FLEX_TEMPLATE_REQUEST(launchParameter=Templates.FLEX_TEMPLATE_PARAMETER(jobName=template_args.job_name, containerSpecGcsPath=template_args.gcs_location, environment=Templates.FLEX_TEMPLATE_ENVIRONMENT(serviceAccountEmail=template_args.service_account_email, maxWorkers=template_args.max_workers, numWorkers=template_args.num_workers, network=template_args.network, subnetwork=template_args.subnetwork, machineType=template_args.worker_machine_type, tempLocation=template_args.temp_location if template_args.temp_location else template_args.staging_location, stagingLocation=template_args.staging_location, kmsKeyName=template_args.kms_key_name, ipConfiguration=ip_configuration, workerRegion=template_args.worker_region, workerZone=template_args.worker_zone, enableStreamingEngine=template_args.enable_streaming_engine, flexrsGoal=flexrs_goal, additionalExperiments=template_args.additional_experiments if template_args.additional_experiments else [], additionalUserLabels=Templates.FLEX_TEMPLATE_USER_LABELS_VALUE(additionalProperties=user_labels_list) if user_labels_list else None), update=streaming_update, transformNameMappings=transform_mappings, parameters=Templates.FLEX_TEMPLATE_PARAMETERS_VALUE(additionalProperties=params_list) if params_list else None))
    request = GetMessagesModule().DataflowProjectsLocationsFlexTemplatesLaunchRequest(projectId=template_args.project_id or GetProject(), location=region_id, launchFlexTemplateRequest=body)
    try:
        return Templates.GetFlexTemplateService().Launch(request)
    except apitools_exceptions.HttpError as error:
        raise exceptions.HttpException(error)