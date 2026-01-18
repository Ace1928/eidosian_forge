from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core.util import files
def GetKeyFromArgs(args):
    if args.key_file:
        key = files.ReadFileContents(args.key_file)
    else:
        key = args.key
    return key