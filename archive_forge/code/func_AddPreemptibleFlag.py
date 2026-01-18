from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddPreemptibleFlag(parser):
    return parser.add_argument('--preemptible', required=False, action='store_true', default=False, help='      Create a preemptible Cloud TPU, instead of a normal (non-preemptible) Cloud TPU. A\n        preemptible Cloud TPU costs less per hour, but the Cloud TPU service can stop/terminate\n        the node at any time.\n      ')