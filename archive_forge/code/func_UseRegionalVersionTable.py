from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.secrets import args as secrets_args
def UseRegionalVersionTable(parser: parser_arguments.ArgumentInterceptor, api_version='v1'):
    """Table format to display regional secret versions.

  Args:
    parser: arguments interceptor
    api_version: api version to be included in resource name
  """
    parser.display_info.AddFormat(_VERSION_TABLE)
    parser.display_info.AddTransforms(_VERSION_STATE_TRANSFORMS)
    secrets_args.MakeGetUriFunc('secretmanager.projects.locations.secrets.versions', api_version=api_version)