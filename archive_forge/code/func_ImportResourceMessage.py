from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core.console import console_io
def ImportResourceMessage(self, yaml_file, message_name):
    """Import a messages class instance typed by name from a YAML file."""
    data = console_io.ReadFromFileOrStdin(yaml_file, binary=False)
    message_type = self.GetMessage(message_name)
    return export_util.Import(message_type=message_type, stream=data)