from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class SessionTemplates(base.Group):
    """Create and manage Dataproc session templates.

  Create and manage Dataproc session templates.

  ## EXAMPLES

  To see the list of all session templates, run:

    $ {command} list

  To view the details of a session template, run:

    $ {command} describe my-template

  To view just the non-output only fields of a session template, run:

    $ {command} export my-template --destination template-file.yaml

  To create or update a session template, run:

    $ {command} import my-template --source template-file.yaml

  To delete a session template, run:

    $ {command} delete my-template
  """
    pass