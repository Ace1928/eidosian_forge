from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class Modules(base.Group):
    """Manage Cloud SCC modules settings."""
    category = base.SECURITY_CATEGORY