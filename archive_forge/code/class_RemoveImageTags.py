from oslo_log import log as logging
from saharaclient.osc.v1 import images as images_v1
class RemoveImageTags(images_v1.RemoveImageTags):
    """Remove image tags"""
    log = logging.getLogger(__name__ + '.RemoveImageTags')