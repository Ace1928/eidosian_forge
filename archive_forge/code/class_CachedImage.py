from openstack import exceptions
from openstack import resource
from openstack import utils
class CachedImage(resource.Resource):
    image_id = resource.Body('image_id')
    hits = resource.Body('hits')
    last_accessed = resource.Body('last_accessed')
    last_modified = resource.Body('last_modified')
    size = resource.Body('size')