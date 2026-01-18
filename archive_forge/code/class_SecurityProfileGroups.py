from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class SecurityProfileGroups(base.Group):
    """Manage Network Security - Security Profile Groups.

  Manage Network Security - Security Profile Groups.

  ## EXAMPLES

  To create a Security Profile Group with the name `my-security-profile-group`
  (Either a fully specified path or `--location` and `--organization` flags for
  SPG should be specified), `--threat-prevention-profile` `my-security-profile`
  and optional description as `optional description`, run:

      $ {command} create my-security-profile-group --organization=1234
      --location=global
      --threat-prevention-profile=`organizations/1234/locations/global/securityProfiles/my-security-profile`
      --description='optional description'

  To delete an Security Profile Group called `my-security-profile-group` (Either
  a fully specified path or `--location` and `--organization` flags for SPG
  should be specified) run:

      $ {command} delete my-security-profile-group --organization=1234
      --location=global

  To show details of a Security Profile Group named `my-security-profile-group`
  (Either a fully specified path or `--location` and `--organization` flags for
  SPG should be specified) run:

      $ {command} describe my-security-profile-group --organization=1234
      --location=global

  To list Security Profile Groups in specified location and organization, run:

      $ {command} list --location=global

  To update an SPG with new Threat prevention profile `my-new-security-profile`
  (Either a fully specified path or `--location` and `--organization` flags for
  SPG should be specified), run:

      $ {command} update my-security-profile-group --organization=1234
      --location=global
      --threat-prevention-profile=`organizations/1234/locations/global/securityProfiles/my-new-security-profile`
      --description='New Security Profile of type threat prevention'
  """
    category = base.NETWORK_SECURITY_CATEGORY