from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
def GetHostProjectFlag(verb):
    """Return the --host-project flag for XPN commands."""
    arg = base.Argument('--host-project', required=True, help='The XPN host to {verb}'.format(verb=verb))
    return arg