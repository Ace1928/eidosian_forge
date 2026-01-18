from libcloud.utils.py3 import ET, tostring
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.common.abiquo import AbiquoResponse, AbiquoConnection, get_href
from libcloud.compute.types import Provider, LibcloudError
def _get_locations(self, location=None):
    """
        Returns the locations as a generator.
        """
    if location is not None:
        yield location
    else:
        yield from self.list_locations()