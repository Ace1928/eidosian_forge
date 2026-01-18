from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import enum
import functools
import re
from googlecloudsdk.api_lib.compute import filter_rewrite
from googlecloudsdk.api_lib.compute.regions import service as regions_service
from googlecloudsdk.api_lib.compute.zones import service as zones_service
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute import scope_prompter
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.util import text
import six
def AddSourceDiskCsekKeyArg(parser):
    spec = {'disk': str, 'csek-key-file': str}
    parser.add_argument('--source-disk-csek-key', type=arg_parsers.ArgDict(spec=spec), action='append', metavar='PROPERTY=VALUE', help='\n              Customer-supplied encryption key of the disk attached to the\n              source instance. Required if the source disk is protected by\n              a customer-supplied encryption key. This flag can be repeated to\n              specify multiple attached disks.\n\n              *disk*::: URL of the disk attached to the source instance.\n              This can be a full or   valid partial URL\n\n              *csek-key-file*::: path to customer-supplied encryption key.\n            ')