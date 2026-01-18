from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute.instances.create import utils as create_utils
from googlecloudsdk.command_lib.compute import resource_manager_tags_utils
from googlecloudsdk.command_lib.compute import secure_tags_utils
from googlecloudsdk.command_lib.compute.resource_policies import util as maintenance_util
from googlecloudsdk.command_lib.util.apis import arg_utils
import six
class SupportedFeatures:
    """Simple dataclass to hold status of supported features in Bulk."""

    def __init__(self, support_nvdimm, support_public_dns, support_erase_vss, support_min_node_cpu, support_source_snapshot_csek, support_image_csek, support_confidential_compute, support_post_key_revocation_action_type, support_rsa_encrypted, deprecate_maintenance_policy, support_create_disk_snapshots, support_boot_snapshot_uri, support_display_device, support_local_ssd_size, support_secure_tags, support_host_error_timeout_seconds, support_numa_node_count, support_visible_core_count, support_max_run_duration, support_local_ssd_recovery_timeout, support_enable_target_shape, support_confidential_compute_type, support_confidential_compute_type_tdx, support_max_count_per_zone, support_performance_monitoring_unit, support_custom_hostnames, support_specific_then_x_affinity, support_watchdog_timer):
        self.support_rsa_encrypted = support_rsa_encrypted
        self.support_secure_tags = support_secure_tags
        self.support_erase_vss = support_erase_vss
        self.support_public_dns = support_public_dns
        self.support_nvdimm = support_nvdimm
        self.support_min_node_cpu = support_min_node_cpu
        self.support_source_snapshot_csek = support_source_snapshot_csek
        self.support_image_csek = support_image_csek
        self.support_confidential_compute = support_confidential_compute
        self.support_post_key_revocation_action_type = support_post_key_revocation_action_type
        self.deprecate_maintenance_policy = deprecate_maintenance_policy
        self.support_create_disk_snapshots = support_create_disk_snapshots
        self.support_boot_snapshot_uri = support_boot_snapshot_uri
        self.support_display_device = support_display_device
        self.support_local_ssd_size = support_local_ssd_size
        self.support_host_error_timeout_seconds = support_host_error_timeout_seconds
        self.support_numa_node_count = support_numa_node_count
        self.support_visible_core_count = support_visible_core_count
        self.support_max_run_duration = support_max_run_duration
        self.support_enable_target_shape = support_enable_target_shape
        self.support_confidential_compute_type = support_confidential_compute_type
        self.support_confidential_compute_type_tdx = support_confidential_compute_type_tdx
        self.support_max_count_per_zone = support_max_count_per_zone
        self.support_local_ssd_recovery_timeout = support_local_ssd_recovery_timeout
        self.support_performance_monitoring_unit = support_performance_monitoring_unit
        self.support_custom_hostnames = support_custom_hostnames
        self.support_specific_then_x_affinity = support_specific_then_x_affinity
        self.support_watchdog_timer = support_watchdog_timer