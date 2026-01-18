from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import import_util
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def AddBakImportFlags(parser, filetype, gz_supported=True, user_supported=True):
    """Adds the base flags for importing data for bak import.

  Args:
    parser: An argparse parser that you can use to add arguments that go on the
      command line after this command. Positional arguments are allowed.
    filetype: String, description of the file type being imported.
    gz_supported: Boolean, if True then .gz compressed files are supported.
    user_supported: Boolean, if True then a Postgres user can be specified.
  """
    base.ASYNC_FLAG.AddToParser(parser)
    flags.AddInstanceArgument(parser)
    uri_help_text = 'Path to the {filetype} file in Google Cloud Storage from which the import is made. The URI is in the form `gs://bucketName/fileName`.'
    if gz_supported:
        uri_help_text = uri_help_text + ' Compressed gzip files (.gz) are also supported.'
    flags.AddBakImportUriArgument(parser, uri_help_text.format(filetype=filetype))
    if user_supported:
        flags.AddUser(parser, 'PostgreSQL user for this import operation.')