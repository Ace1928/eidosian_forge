import datetime
import sys
from functools import partial
from optparse import OptionGroup, OptionParser, OptionValueError
from subunit import make_stream_binary
from iso8601 import UTC
from subunit.v2 import StreamResultToBytes
Parse arguments from the command line.

    If specified, args must be a list of strings, similar to sys.argv[1:].

    ParserClass may be specified to override the class we use to parse the
    command-line arguments. This is useful for testing.
    