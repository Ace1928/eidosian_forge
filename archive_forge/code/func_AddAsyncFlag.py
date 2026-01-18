from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddAsyncFlag(parser):
    """Adds --async flag."""
    base.ASYNC_FLAG.AddToParser(parser)