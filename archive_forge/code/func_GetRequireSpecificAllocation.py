from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
def GetRequireSpecificAllocation():
    help_text = '  Indicates whether the reservation can be consumed by VMs with "any reservation"\n  defined. If enabled, then only VMs that target this reservation by name using\n  `--reservation-affinity=specific` can consume from this reservation.\n  '
    return base.Argument('--require-specific-reservation', action='store_true', help=help_text)