from oslo_log import log as logging
from saharaclient.osc.v1 import images as images_v1
class RegisterImage(images_v1.RegisterImage):
    """Register an image"""
    log = logging.getLogger(__name__ + '.RegisterImage')