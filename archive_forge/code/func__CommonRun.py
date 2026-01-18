from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataflow import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataflow import dataflow_util
from googlecloudsdk.command_lib.dataflow import job_utils
from googlecloudsdk.core import properties
def _CommonRun(args):
    """Runs the command.

  Args:
    args: The arguments that were provided to this command invocation.

  Returns:
    A Job message.
  """
    arguments = apis.TemplateArguments(project_id=properties.VALUES.core.project.Get(required=True), region_id=dataflow_util.GetRegion(args), job_name=args.job_name, gcs_location=args.gcs_location, zone=args.zone, max_workers=args.max_workers, num_workers=args.num_workers, network=args.network, subnetwork=args.subnetwork, worker_machine_type=args.worker_machine_type, staging_location=args.staging_location, kms_key_name=args.dataflow_kms_key, disable_public_ips=properties.VALUES.dataflow.disable_public_ips.GetBool(), parameters=args.parameters, service_account_email=args.service_account_email, worker_region=args.worker_region, worker_zone=args.worker_zone, enable_streaming_engine=properties.VALUES.dataflow.enable_streaming_engine.GetBool(), additional_experiments=args.additional_experiments)
    return apis.Templates.Create(arguments)