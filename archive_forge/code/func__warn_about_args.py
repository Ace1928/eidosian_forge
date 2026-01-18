import boto
import os
import sys
import textwrap
from boto.s3.deletemarker import DeleteMarker
from boto.exception import BotoClientError
from boto.exception import InvalidUriError
def _warn_about_args(self, function_name, **args):
    for arg in args:
        if args[arg]:
            sys.stderr.write('Warning: %s ignores argument: %s=%s\n' % (function_name, arg, str(args[arg])))