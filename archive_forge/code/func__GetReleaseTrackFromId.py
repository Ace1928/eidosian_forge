from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.calliope import markdown
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import properties
import six
def _GetReleaseTrackFromId(release_id):
    """Returns the base.ReleaseTrack for release_id."""
    if release_id == 'INTERNAL':
        release_id = 'GA'
    return base.ReleaseTrack.FromId(release_id)