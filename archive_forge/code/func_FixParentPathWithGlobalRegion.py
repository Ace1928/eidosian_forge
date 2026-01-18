from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def FixParentPathWithGlobalRegion(region: str) -> str:
    """Returns projects/$project/location/$location parent path based on region."""
    if region is not None:
        return region.RelativeName()
    project = properties.VALUES.core.project.Get(required=True)
    return 'projects/{}/locations/global'.format(project)