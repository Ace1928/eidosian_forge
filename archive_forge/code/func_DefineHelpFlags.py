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
def DefineHelpFlags():
    """Register help flags. Idempotent."""
    global _define_help_flags_called
    if not _define_help_flags_called:
        flags.DEFINE_flag(HelpFlag())
        flags.DEFINE_flag(HelpXMLFlag())
        flags.DEFINE_flag(HelpshortFlag())
        flags.DEFINE_flag(BuildDataFlag())
        _define_help_flags_called = True