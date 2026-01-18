from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import xml.etree.cElementTree as element_tree
from googlecloudsdk.command_lib.metastore import parsers
from googlecloudsdk.core import properties
def LoadScheduledBackupConfigsFromJsonFile(file_contents):
    """Convert Input JSON file into scheduled backup configurations map.

  Args:
    file_contents: The JSON file contents of the file containing the scheduled
      backup configurations.

  Returns:
    The scheduled backup configuration mapping with key and value.
  """
    try:
        scheduled_backup_configs = json.loads(file_contents)
        config = {}
        if 'enabled' in scheduled_backup_configs:
            config['enabled'] = scheduled_backup_configs.pop('enabled')
        if config.get('enabled', False):
            if 'cron_schedule' not in scheduled_backup_configs:
                raise ValueError('Missing required field: cron_schedule')
            if 'backup_location' not in scheduled_backup_configs:
                raise ValueError('Missing required field: backup_location')
        config['cron_schedule'] = scheduled_backup_configs.get('cron_schedule')
        config['backup_location'] = scheduled_backup_configs.get('backup_location')
        config['time_zone'] = scheduled_backup_configs.get('time_zone', 'UTC')
        return config
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f'Invalid scheduled backup configuration JSON data: {e}')