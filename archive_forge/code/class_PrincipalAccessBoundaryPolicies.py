from googlecloudsdk.calliope import base
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class PrincipalAccessBoundaryPolicies(base.Group):
    """Manage PrincipalAccessBoundaryPolicy instances."""