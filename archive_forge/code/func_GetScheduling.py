from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute import zone_utils
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.command_lib.compute.sole_tenancy import util as sole_tenancy_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core.util import times
import six
def GetScheduling(args, client, skip_defaults, support_node_affinity=False, support_min_node_cpu=True, support_node_project=False, support_host_error_timeout_seconds=False, support_max_run_duration=False, support_local_ssd_recovery_timeout=False, support_graceful_shutdown=False):
    """Generate a Scheduling Message or None based on specified args."""
    node_affinities = None
    if support_node_affinity:
        node_affinities = sole_tenancy_util.GetSchedulingNodeAffinityListFromArgs(args, client.messages, support_node_project)
    min_node_cpu = None
    if support_min_node_cpu:
        min_node_cpu = args.min_node_cpu
    location_hint = None
    if hasattr(args, 'location_hint'):
        location_hint = args.location_hint
    freeze_duration = None
    if hasattr(args, 'maintenance_freeze_duration') and args.IsSpecified('maintenance_freeze_duration'):
        freeze_duration = args.maintenance_freeze_duration
    maintenance_interval = None
    if hasattr(args, 'maintenance_interval') and args.IsSpecified('maintenance_interval'):
        maintenance_interval = args.maintenance_interval
    provisioning_model = None
    if hasattr(args, 'provisioning_model') and args.IsSpecified('provisioning_model'):
        provisioning_model = args.provisioning_model
    instance_termination_action = None
    if hasattr(args, 'instance_termination_action') and args.IsSpecified('instance_termination_action'):
        instance_termination_action = args.instance_termination_action
    host_error_timeout_seconds = None
    if support_host_error_timeout_seconds and hasattr(args, 'host_error_timeout_seconds'):
        host_error_timeout_seconds = args.host_error_timeout_seconds
    max_run_duration = None
    if support_max_run_duration and hasattr(args, 'max_run_duration'):
        max_run_duration = args.max_run_duration
    local_ssd_recovery_timeout = None
    if support_local_ssd_recovery_timeout and hasattr(args, 'local_ssd_recovery_timeout') and args.IsSpecified('local_ssd_recovery_timeout'):
        local_ssd_recovery_timeout = args.local_ssd_recovery_timeout
    graceful_shutdown = ExtractGracefulShutdownFromArgs(args, support_graceful_shutdown)
    termination_time = None
    if support_max_run_duration and hasattr(args, 'termination_time'):
        termination_time = args.termination_time
    discard_local_ssds_at_termination_timestamp = None
    if support_max_run_duration and hasattr(args, 'discard_local_ssds_at_termination_timestamp'):
        discard_local_ssds_at_termination_timestamp = args.discard_local_ssds_at_termination_timestamp
    restart_on_failure = None
    if not skip_defaults or args.IsKnownAndSpecified('restart_on_failure'):
        restart_on_failure = args.restart_on_failure
    availability_domain = None
    if args.IsKnownAndSpecified('availability_domain') and hasattr(args, 'availability_domain'):
        availability_domain = args.availability_domain
    if skip_defaults and (not IsAnySpecified(args, 'instance_termination_action', 'maintenance_policy', 'preemptible', 'provisioning_model')) and (not restart_on_failure) and (not node_affinities) and (not max_run_duration) and (not termination_time) and (not freeze_duration) and (not host_error_timeout_seconds) and (not maintenance_interval) and (not local_ssd_recovery_timeout) and (not graceful_shutdown):
        return None
    return CreateSchedulingMessage(messages=client.messages, maintenance_policy=args.maintenance_policy, preemptible=args.preemptible, restart_on_failure=restart_on_failure, node_affinities=node_affinities, min_node_cpu=min_node_cpu, location_hint=location_hint, maintenance_freeze_duration=freeze_duration, maintenance_interval=maintenance_interval, provisioning_model=provisioning_model, instance_termination_action=instance_termination_action, host_error_timeout_seconds=host_error_timeout_seconds, max_run_duration=max_run_duration, termination_time=termination_time, local_ssd_recovery_timeout=local_ssd_recovery_timeout, availability_domain=availability_domain, graceful_shutdown=graceful_shutdown, discard_local_ssds_at_termination_timestamp=discard_local_ssds_at_termination_timestamp)