from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddQuicOverrideUpdateArgs(parser):
    """Adds parser arguments for update related to QuicOverride."""
    AddQuicOverrideCreateArgs(parser, default=None)