from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.service_extensions import wasm_plugin_api
from googlecloudsdk.api_lib.service_extensions import wasm_plugin_version_api
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.service_extensions import flags
from googlecloudsdk.command_lib.service_extensions import util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
def GetPluginConfigData(args):
    return args.plugin_config or args.plugin_config_file