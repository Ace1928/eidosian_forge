from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from typing import Any
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddMinReadyArg(parser):
    parser.add_argument('--min-ready', type=arg_parsers.Duration(lower_bound='0s'), help='Minimum time for which a newly created instance should be ready to be considered available. For example `10s` for 10 seconds. See $ gcloud topic datetimes for information on duration formats.')