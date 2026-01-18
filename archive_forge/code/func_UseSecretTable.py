from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.secrets import args as secrets_args
def UseSecretTable(parser: parser_arguments.ArgumentInterceptor):
    """Table format to display secrets.

  Args:
    parser: arguments interceptor
  """
    parser.display_info.AddFormat(_SECRET_TABLE)
    parser.display_info.AddTransforms(_SECRET_TRANSFORMS)
    parser.display_info.AddUriFunc(lambda r: secrets_args.ParseSecretRef(r.name).SelfLink())