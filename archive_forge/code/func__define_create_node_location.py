from libcloud.utils.py3 import ET, tostring
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.common.abiquo import AbiquoResponse, AbiquoConnection, get_href
from libcloud.compute.types import Provider, LibcloudError
def _define_create_node_location(self, image, location):
    """
        Search for a location where to create the node.

        Based on 'create_node' **kwargs argument, decide in which
        location will be created.
        """
    if not image:
        error = "'image' parameter is mandatory"
        raise LibcloudError(error, self)
    if location:
        if location not in self.list_locations():
            raise LibcloudError('Location does not exist')
    loc = None
    target_loc = None
    for candidate_loc in self._get_locations(location):
        link_vdc = self.connection.cache['locations'][candidate_loc]
        hdr_vdc = {'Accept': self.VDC_MIME_TYPE}
        e_vdc = self.connection.request(link_vdc, headers=hdr_vdc).object
        for img in self.list_images(candidate_loc):
            if img.id == image.id:
                loc = e_vdc
                target_loc = candidate_loc
                break
    if loc is None:
        error = 'The image can not be used in any location'
        raise LibcloudError(error, self)
    return (loc, target_loc)