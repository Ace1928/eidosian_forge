from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions
import six
def AddServiceConfigPositionalArg(self, include_app_engine_docs=False):
    """_AddFlag for service_config, which has two possible help strings.

    Args:
      include_app_engine_docs: Add paragraph that says app.yaml is allowed.
    """
    help_text = 'service.yaml filename override. Defaults to the first file matching ```*service.dev.yaml``` then ```*service.yaml```, if any exist. This path is relative to the --source dir.'
    if include_app_engine_docs:
        help_text += '\nAn App Engine config path (typically ```app.yaml```) may also be provided here, and we will build with a Cloud Native Computing Foundation Buildpack builder selected from gcr.io/gae-runtimes/buildpacks, according to the App Engine ```runtime``` specified in app.yaml.'
    self._AddFlag('service_config', metavar='SERVICE_CONFIG', nargs='?', help=help_text)