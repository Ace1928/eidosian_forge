import operator
import os
import time
import uuid
from keystoneauth1 import discover
import openstack.config
from openstack import connection
from openstack.tests import base
def _set_user_cloud(self, **kwargs):
    user_config = self.config.get_one(cloud=self._demo_name, **kwargs)
    self.user_cloud = connection.Connection(config=user_config)
    _disable_keep_alive(self.user_cloud)
    if self._demo_name_alt:
        user_config_alt = self.config.get_one(cloud=self._demo_name_alt, **kwargs)
        self.user_cloud_alt = connection.Connection(config=user_config_alt)
        _disable_keep_alive(self.user_cloud_alt)
    else:
        self.user_cloud_alt = None