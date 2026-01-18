from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.error_reporting import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.error_reporting import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def GetMessage(self, args):
    """Get error message.

    Args:
      args: the arguments for the command

    Returns:
      error_message read from error file or provided inline

    Raises:
      CannotOpenFileError: When there is a problem with reading the file
    """
    error_message = ''
    if args.message_file:
        try:
            error_message = files.ReadFileContents(args.message_file)
        except files.Error as e:
            raise exceptions.CannotOpenFileError(args.message_file, e)
    elif args.message:
        error_message = args.message
    return error_message