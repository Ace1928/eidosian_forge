import getopt
import sys
from gslib.exception import CommandException
def GetArgumentsAndOptions():
    """Gets the list of arguments and options from the command input.

  Returns:
    The return value consists of two elements: the first is a list of (option,
    value) pairs; the second is the list of program arguments left after the
    option list was stripped (this is a trailing slice of the first argument).
  """
    try:
        return getopt.getopt(sys.argv[1:], 'dDvo:?h:i:u:mq', ['debug', 'detailedDebug', 'version', 'option', 'help', 'header', 'impersonate-service-account=', 'multithreaded', 'quiet', 'testexceptiontraces', 'trace-token=', 'perf-trace-token='])
    except getopt.GetoptError as e:
        raise CommandException(e.msg)