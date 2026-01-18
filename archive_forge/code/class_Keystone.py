import glanceclient
from keystoneauth1 import loading
from keystoneauth1 import session
import os
import os_client_config
from tempest.lib.cli import base
class Keystone(object):

    def __init__(self, **kwargs):
        loader = loading.get_plugin_loader('password')
        auth = loader.load_from_options(**kwargs)
        self.session = session.Session(auth=auth)