import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def EncodeRspFileList(args, quote_cmd):
    """Process a list of arguments using QuoteCmdExeArgument."""
    if not args:
        return ''
    if args[0].startswith('call '):
        call, program = args[0].split(' ', 1)
        program = call + ' ' + os.path.normpath(program)
    else:
        program = os.path.normpath(args[0])
    return program + ' ' + ' '.join((QuoteForRspFile(arg, quote_cmd) for arg in args[1:]))