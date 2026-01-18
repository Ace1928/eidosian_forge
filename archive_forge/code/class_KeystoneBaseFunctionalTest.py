import operator
import os
import time
import uuid
from keystoneauth1 import discover
import openstack.config
from openstack import connection
from openstack.tests import base
class KeystoneBaseFunctionalTest(BaseFunctionalTest):

    def setUp(self):
        super(KeystoneBaseFunctionalTest, self).setUp()
        use_keystone_v2 = os.environ.get('OPENSTACKSDK_USE_KEYSTONE_V2', False)
        if use_keystone_v2:
            self._set_operator_cloud(interface='admin')