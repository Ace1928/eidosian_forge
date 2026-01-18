from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.secrets import args as secrets_args
def UseLocationTable(parser: parser_arguments.ArgumentInterceptor, api_version: str='v1'):
    """Table format to display locations.

  Args:
    parser: arguments interceptor
    api_version: api version to be included in resource name
  """
    parser.display_info.AddFormat(_LOCATION_TABLE)
    parser.display_info.AddUriFunc(secrets_args.MakeGetUriFunc('secretmanager.projects.locations', api_version=api_version))