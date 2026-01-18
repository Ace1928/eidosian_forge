from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import filter_rewrite
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.compute.instances.bulk import flags as bulk_flags
from googlecloudsdk.command_lib.compute.instances.bulk import util as bulk_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _CreateRequests(self, args, holder, compute_client, resource_parser, project, location, scope):
    supported_features = bulk_util.SupportedFeatures(self._support_nvdimm, self._support_public_dns, self._support_erase_vss, self._support_min_node_cpu, self._support_source_snapshot_csek, self._support_image_csek, self._support_confidential_compute, self._support_post_key_revocation_action_type, self._support_rsa_encrypted, self._deprecate_maintenance_policy, self._support_create_disk_snapshots, self._support_boot_snapshot_uri, self._support_display_device, self._support_local_ssd_size, self._support_secure_tags, self._support_host_error_timeout_seconds, self._support_numa_node_count, self._support_visible_core_count, self._support_max_run_duration, self._support_local_ssd_recovery_timeout, self._support_enable_target_shape, self._support_confidential_compute_type, self._support_confidential_compute_type_tdx, self._support_max_count_per_zone, self._support_performance_monitoring_unit, self._support_custom_hostnames, self._support_specific_then_x_affinity, self._support_watchdog_timer)
    bulk_instance_resource = bulk_util.CreateBulkInsertInstanceResource(args, holder, compute_client, resource_parser, project, location, scope, self.SOURCE_INSTANCE_TEMPLATE, supported_features)
    if scope == compute_scopes.ScopeEnum.ZONE:
        instance_service = compute_client.apitools_client.instances
        request_message = compute_client.messages.ComputeInstancesBulkInsertRequest(bulkInsertInstanceResource=bulk_instance_resource, project=project, zone=location)
    elif scope == compute_scopes.ScopeEnum.REGION:
        instance_service = compute_client.apitools_client.regionInstances
        request_message = compute_client.messages.ComputeRegionInstancesBulkInsertRequest(bulkInsertInstanceResource=bulk_instance_resource, project=project, region=location)
    return (instance_service, request_message)