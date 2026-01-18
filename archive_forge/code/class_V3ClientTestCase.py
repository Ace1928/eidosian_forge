import testtools
from keystoneclient import client
import os_client_config
class V3ClientTestCase(ClientTestCase):
    version = '3'