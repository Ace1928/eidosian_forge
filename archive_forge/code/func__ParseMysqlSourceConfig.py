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
def _ParseMysqlSourceConfig(self, mysql_source_config_file, release_track):
    """Parses a mysql_sorce_config into the MysqlSourceConfig message."""
    if release_track == base.ReleaseTrack.BETA:
        return self._ParseMysqlSourceConfigBeta(mysql_source_config_file, release_track)
    return util.ParseMessageAndValidateSchema(mysql_source_config_file, 'MysqlSourceConfig', self._messages.MysqlSourceConfig)