from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import xml.etree.cElementTree as element_tree
from googlecloudsdk.command_lib.metastore import parsers
from googlecloudsdk.core import properties
def _GenerateUpdateMask(args):
    """Constructs updateMask for patch requests.

  Args:
    args: The parsed args namespace from CLI.

  Returns:
    String containing update mask for patch request.
  """
    hive_metastore_configs = 'hive_metastore_config.config_overrides'
    labels = 'labels'
    arg_name_to_field = {'--port': 'port', '--tier': 'tier', '--instance-size': 'scaling_config.instance_size', '--scaling-factor': 'scaling_config.scaling_factor', '--autoscaling-enabled': 'scaling_config.autoscaling_config.autoscaling_enabled', '--min-scaling-factor': 'scaling_config.autoscaling_config.limit_config.min_scaling_factor', '--max-scaling-factor': 'scaling_config.autoscaling_config.limit_config.max_scaling_factor', '--update-hive-metastore-configs-from-file': 'hive_metastore_config.config_overrides', '--clear-hive-metastore-configs': hive_metastore_configs, '--clear-labels': labels, '--kerberos-principal': 'hive_metastore_config.kerberos_config.principal', '--keytab': 'hive_metastore_config.kerberos_config.keytab', '--krb5-config': 'hive_metastore_config.kerberos_config.krb5_config_gcs_uri', '--maintenance-window-day': 'maintenance_window', '--maintenance-window-hour': 'maintenance_window', '--data-catalog-sync': 'metadataIntegration.dataCatalogConfig.enabled', '--no-data-catalog-sync': 'metadataIntegration.dataCatalogConfig.enabled', '--endpoint-protocol': 'hive_metastore_config.endpoint_protocol', '--add-auxiliary-versions': 'hive_metastore_config.auxiliary_versions', '--update-auxiliary-versions-from-file': 'hive_metastore_config.auxiliary_versions', '--clear-auxiliary-versions': 'hive_metastore_config.auxiliary_versions', '--scheduled-backup-configs-from-file': 'scheduled_backup', '--enable-scheduled-backup': 'scheduled_backup', '--no-enable-scheduled-backup': 'scheduled_backup.enabled', '--scheduled-backup-cron': 'scheduled_backup', '--scheduled-backup-location': 'scheduled_backup'}
    update_mask = set()
    for arg_name in set(args.GetSpecifiedArgNames()).intersection(arg_name_to_field):
        update_mask.add(arg_name_to_field[arg_name])
    hive_metastore_configs_update_mask_prefix = hive_metastore_configs + '.'
    if hive_metastore_configs not in update_mask:
        if args.update_hive_metastore_configs:
            for key in args.update_hive_metastore_configs:
                update_mask.add(hive_metastore_configs_update_mask_prefix + key)
        if args.remove_hive_metastore_configs:
            for key in args.remove_hive_metastore_configs:
                update_mask.add(hive_metastore_configs_update_mask_prefix + key)
    labels_update_mask_prefix = labels + '.'
    if labels not in update_mask:
        if args.update_labels:
            for key in args.update_labels:
                update_mask.add(labels_update_mask_prefix + key)
        if args.remove_labels:
            for key in args.remove_labels:
                update_mask.add(labels_update_mask_prefix + key)
    return ','.join(sorted(update_mask))