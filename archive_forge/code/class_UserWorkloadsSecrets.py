from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class UserWorkloadsSecrets(base.Group):
    """Create and manage user workloads Secrets of environment.

  The {command} command group manages user workloads Secrets of Cloud Composer
  environments.
  """