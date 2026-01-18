from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class DevKits(base.Group):
    """Top level command to interact with Kuberun Development Kits.

  This set of commands can be used to list available Development Kits and view
  details of a specific Development Kit.
  """
    detailed_help = {'EXAMPLES': '          To list all available Development Kits, run:\n\n            $ {command} devkits list\n      '}