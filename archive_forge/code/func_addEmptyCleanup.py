import operator
import os
import time
import uuid
from keystoneauth1 import discover
import openstack.config
from openstack import connection
from openstack.tests import base
def addEmptyCleanup(self, func, *args, **kwargs):

    def cleanup():
        result = func(*args, **kwargs)
        self.assertIsNone(result)
    self.addCleanup(cleanup)