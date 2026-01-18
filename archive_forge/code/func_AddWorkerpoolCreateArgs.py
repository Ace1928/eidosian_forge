from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddWorkerpoolCreateArgs(parser, release_track):
    """Set up all the argparse flags for creating a workerpool.

  Args:
    parser: An argparse.ArgumentParser-like object.
    release_track: A base.ReleaseTrack-like object.

  Returns:
    The parser argument with workerpool flags added in.
  """
    return AddWorkerpoolArgs(parser, release_track, update=False)