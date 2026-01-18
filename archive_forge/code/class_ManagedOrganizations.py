from googlecloudsdk.calliope import base
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class ManagedOrganizations(base.Group):
    """Manage ManagedOrganization resources.

  This allows resellers to manage Cloud Resource Manager organizations
  on behalf of their customers.
  """