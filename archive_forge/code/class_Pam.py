from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class Pam(base.Group):
    """Manage Privileged Access Manager (PAM) Entitlements and Grants.

  The gcloud pam command group lets you manage Privileged Access Manager
  (PAM) Entitlements and Grants.

  ## EXAMPLES

  To check the PAM onboarding status for a project `sample-project` and
  location `global`, run:

      $ {command} check-onboarding-status --project=sample-project
      --location=global

  To check the PAM onboarding status for a folder `sample-folder` and
  location `global`, run:

      $ {command} check-onboarding-status --folder=sample-folder
      --location=global

  To check the PAM onboarding status for an organization
  `sample-organization` and location `global`, run:

      $ {command} check-onboarding-status --organization=sample-organization
      --location=global

  """