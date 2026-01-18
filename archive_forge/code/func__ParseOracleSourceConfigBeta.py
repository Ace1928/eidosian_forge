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
def _ParseOracleSourceConfigBeta(self, oracle_source_config_file, release_track):
    """Parses a oracle_sorce_config into the OracleSourceConfig message."""
    data = console_io.ReadFromFileOrStdin(oracle_source_config_file, binary=False)
    try:
        oracle_source_config_head_data = yaml.load(data)
    except yaml.YAMLParseError as e:
        raise ds_exceptions.ParseError('Cannot parse YAML:[{0}]'.format(e))
    oracle_sorce_config_data_object = oracle_source_config_head_data.get('oracle_source_config')
    oracle_source_config = oracle_sorce_config_data_object if oracle_sorce_config_data_object else oracle_source_config_head_data
    include_objects_raw = oracle_source_config.get(util.GetRDBMSV1alpha1ToV1FieldName('include_objects', release_track), {})
    include_objects_data = util.ParseOracleSchemasListToOracleRdbmsMessage(self._messages, include_objects_raw, release_track)
    exclude_objects_raw = oracle_source_config.get(util.GetRDBMSV1alpha1ToV1FieldName('exclude_objects', release_track), {})
    exclude_objects_data = util.ParseOracleSchemasListToOracleRdbmsMessage(self._messages, exclude_objects_raw, release_track)
    oracle_source_config_msg = self._messages.OracleSourceConfig(includeObjects=include_objects_data, excludeObjects=exclude_objects_data)
    if oracle_source_config.get('max_concurrent_cdc_tasks'):
        oracle_source_config_msg.maxConcurrentCdcTasks = oracle_source_config.get('max_concurrent_cdc_tasks')
    return oracle_source_config_msg