from unittest import mock
import ddt
from oslo_db import exception as db_exc
from oslotest import base
from neutron_lib.callbacks import events
from neutron_lib.callbacks import exceptions
from neutron_lib.callbacks import manager
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import resources
class ObjectWithCallback(object):

    def __init__(self):
        self.counter = 0

    def callback(self, *args, **kwargs):
        self.counter += 1