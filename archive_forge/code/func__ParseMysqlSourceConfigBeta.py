from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.datastream import exceptions as ds_exceptions
from googlecloudsdk.api_lib.datastream import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
def _ParseMysqlSourceConfigBeta(self, mysql_source_config_file, release_track):
    """Parses an old mysql_sorce_config into the MysqlSourceConfig message."""
    data = console_io.ReadFromFileOrStdin(mysql_source_config_file, binary=False)
    try:
        mysql_sorce_config_head_data = yaml.load(data)
    except yaml.YAMLParseError as e:
        raise ds_exceptions.ParseError('Cannot parse YAML:[{0}]'.format(e))
    mysql_sorce_config_data_object = mysql_sorce_config_head_data.get('mysql_source_config')
    mysql_source_config = mysql_sorce_config_data_object if mysql_sorce_config_data_object else mysql_sorce_config_head_data
    include_objects_raw = mysql_source_config.get(util.GetRDBMSV1alpha1ToV1FieldName('include_objects', release_track), {})
    include_objects_data = util.ParseMysqlSchemasListToMysqlRdbmsMessage(self._messages, include_objects_raw, release_track)
    exclude_objects_raw = mysql_source_config.get(util.GetRDBMSV1alpha1ToV1FieldName('exclude_objects', release_track), {})
    exclude_objects_data = util.ParseMysqlSchemasListToMysqlRdbmsMessage(self._messages, exclude_objects_raw, release_track)
    mysql_sourec_config_msg = self._messages.MysqlSourceConfig(includeObjects=include_objects_data, excludeObjects=exclude_objects_data)
    if mysql_source_config.get('max_concurrent_cdc_tasks'):
        mysql_sourec_config_msg.maxConcurrentCdcTasks = mysql_source_config.get('max_concurrent_cdc_tasks')
    return mysql_sourec_config_msg