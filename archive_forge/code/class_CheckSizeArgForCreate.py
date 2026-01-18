import argparse
import collections
import copy
import os
from oslo_utils import strutils
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient import utils
from cinderclient.v3 import availability_zones
class CheckSizeArgForCreate(argparse.Action):

    def __call__(self, parser, args, values, option_string=None):
        if (args.snapshot_id or args.source_volid) is None and values is None:
            if not hasattr(args, 'backup_id') or args.backup_id is None:
                parser.error('Size is a required parameter if snapshot or source volume or backup is not specified.')
        setattr(args, self.dest, values)