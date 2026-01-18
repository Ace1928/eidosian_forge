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
def _GetMySqlProfile(self, args):
    ssl_config = self._GetSslConfig(args)
    return self._messages.MysqlProfile(hostname=args.mysql_hostname, port=args.mysql_port, username=args.mysql_username, password=args.mysql_password, sslConfig=ssl_config)