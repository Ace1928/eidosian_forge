import errno
import os
import pdb
import socket
import stat
import struct
import sys
import time
import traceback
import gflags as flags
class HelpXMLFlag(flags.BooleanFlag):
    """Similar to HelpFlag, but generates output in XML format."""

    def __init__(self):
        flags.BooleanFlag.__init__(self, 'helpxml', False, 'like --help, but generates XML output', allow_override=1)

    def Parse(self, arg):
        if arg:
            flags.FLAGS.WriteHelpInXMLFormat(sys.stdout)
            sys.exit(1)