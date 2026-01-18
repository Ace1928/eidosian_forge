from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddDisablePublicIpAddresses(parser, use_default=True):
    """Adds a --disable-public-ip-addresses flag to the given parser."""
    help_text = '  Default value is false.\n  If set, instances will have no public IP address.'
    parser.add_argument('--disable-public-ip-addresses', action='store_true', default=False if use_default else None, help=help_text)