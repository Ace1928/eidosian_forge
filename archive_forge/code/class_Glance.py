import glanceclient
from keystoneauth1 import loading
from keystoneauth1 import session
import os
import os_client_config
from tempest.lib.cli import base
class Glance(object):

    def __init__(self, keystone, version='2'):
        self.glance = glanceclient.Client(version, session=keystone.session)

    def find(self, image_name):
        for image in self.glance.images.list():
            if image.name == image_name:
                return image
        return None