import getopt
import sys
from gslib.exception import CommandException
Gets the list of arguments and options from the command input.

  Returns:
    The return value consists of two elements: the first is a list of (option,
    value) pairs; the second is the list of program arguments left after the
    option list was stripped (this is a trailing slice of the first argument).
  