from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def WorkloadUpdateMask(self, template_type, args):
    """Given a set of args for the workload, create the required update mask.

    Args:
      template_type: str, the type of the pipeline.
      args: Any, object with args needed for updating a pipeline.

    Returns:
      Update mask.
    """
    update_mask = []
    if template_type == 'flex':
        prefix_string = 'workload.dataflow_flex_template_request.launch_parameter.'
    else:
        prefix_string = 'workload.dataflow_launch_template_request.launch_parameters.'
    if args.template_file_gcs_location:
        if template_type == 'flex':
            update_mask.append(prefix_string + 'container_spec_gcs_path')
        else:
            update_mask.append('workload.dataflow_launch_template_request.gcs_path')
    if args.parameters:
        update_mask.append(prefix_string + 'parameters')
    if args.update:
        update_mask.append(prefix_string + 'update')
    if args.transform_name_mappings:
        if template_type == 'flex':
            update_mask.append(prefix_string + 'transform_name_mappings')
        else:
            update_mask.append(prefix_string + 'transform_name_mapping')
    if args.max_workers:
        update_mask.append(prefix_string + 'environment.max_workers')
    if args.num_workers:
        update_mask.append(prefix_string + 'environment.num_workers')
    if args.dataflow_service_account_email:
        update_mask.append(prefix_string + 'environment.service_account_email')
    if args.temp_location:
        update_mask.append(prefix_string + 'environment.temp_location')
    if args.network:
        update_mask.append(prefix_string + 'environment.network')
    if args.subnetwork:
        update_mask.append(prefix_string + 'environment.subnetwork')
    if args.worker_machine_type:
        update_mask.append(prefix_string + 'environment.machine_type')
    if args.dataflow_kms_key:
        update_mask.append(prefix_string + 'environment.kms_key_name')
    if args.disable_public_ips:
        update_mask.append(prefix_string + 'environment.ip_configuration')
    if args.worker_region:
        update_mask.append(prefix_string + 'environment.worker_region')
    if args.worker_zone:
        update_mask.append(prefix_string + 'environment.worker_zone')
    if args.enable_streaming_engine:
        update_mask.append(prefix_string + 'environment.enable_streaming_engine')
    if args.flexrs_goal:
        if template_type == 'flex':
            update_mask.append(prefix_string + 'environment.flexrs_goal')
    if args.additional_user_labels:
        update_mask.append(prefix_string + 'environment.additional_user_labels')
    if args.additional_experiments:
        update_mask.append(prefix_string + 'environment.additional_experiments')
    return update_mask