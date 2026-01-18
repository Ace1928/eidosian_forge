from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.secrets import args as secrets_args
def RegionalSecretVersionTableUsingArgument(args: parser_extensions.Namespace, api_version: str='v1'):
    """Table format to display regional secrets.

  Args:
    args: arguments interceptor
    api_version: api version to be included in resource name
  """
    args.GetDisplayInfo().AddFormat(_VERSION_TABLE)
    args.GetDisplayInfo().AddTransforms(_VERSION_STATE_TRANSFORMS)
    args.GetDisplayInfo().AddUriFunc(secrets_args.MakeGetUriFunc('secretmanager.projects.locations.secrets.versions', api_version=api_version))