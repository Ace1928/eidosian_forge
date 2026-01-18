from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.datastream import exceptions as ds_exceptions
from googlecloudsdk.api_lib.datastream import util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
def _ParseMySqlProfile(self, data):
    if not data:
        return {}
    ssl_config = self._ParseSslConfig(data)
    return self._messages.MysqlProfile(hostname=data.get('hostname'), port=data.get('port'), username=data.get('username'), password=data.get('password'), sslConfig=ssl_config)