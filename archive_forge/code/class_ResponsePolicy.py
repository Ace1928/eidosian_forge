from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class ResponsePolicy(base.Group):
    """Manage your Cloud DNS response policy.

  ## EXAMPLES

  To create a response policy, run:

    $ {command} create myresponsepolicy --description="My Response Policy" --network=''

  To update a response policy, run:

    $ {command} update myresponsepolicy --description="My Response Policy" --network=''

  To delete a response policy, run:

    $ {command} delete myresponsepolicy

  To view the details of a response policy, run

    $ {command} describe myresponsepolicy

  To see a list of all response policies, run

    $ {command} list
  """
    pass