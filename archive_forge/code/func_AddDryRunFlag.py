from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddDryRunFlag(parser):
    return parser.add_argument('--dry-run', required=False, action='store_true', default=False, help='      Do not make changes; print only what would have happened.\n      ')