from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class Operations(base.Group):
    """Manage Privileged Access Manager (PAM) Long Running Operations.

  The `gcloud pam operations` command group lets you manage Privileged
  Access Manager (PAM) Operations.

  ## EXAMPLES

  To describe an operation with the full name ``OPERATION_NAME'', run:

      $ {command} describe OPERATION_NAME

  To list all operations under a project `sample-project` and location
  `global`, run:

      $ {command} list --project=sample-project --location=global

  To list all operations under a folder `sample-folder` and location
  `global`, run:

      $ {command} list --folder=sample-folder --location=global

  To list all operations under an organization `sample-organization` and
  location `global`, run:

      $ {command} list --organization=sample-organization --location=global

  To delete an operation with the full name ``OPERATION_NAME'', run:

      $ {command} delete OPERATION_NAME

  To poll an operation with the full name ``OPERATION_NAME'', run:

      $ {command} wait OPERATION_NAME

  """