import os
import re
import socket  # for gethostname
import gyp.easy_xml as easy_xml
def _QuoteWin32CommandLineArgs(args):
    new_args = []
    for arg in args:
        if arg.find('"') != -1:
            arg = '""'.join(arg.split('"'))
            arg = '"%s"' % arg
        elif re.search('[ \\t\\n]', arg):
            arg = '"%s"' % arg
        new_args.append(arg)
    return new_args