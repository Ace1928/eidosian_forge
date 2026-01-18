import datetime
import hashlib
import json
import uuid
from openstack.cloud import meta
from openstack.orchestration.util import template_format
from openstack import utils
class FakeMachine:

    def __init__(self, id, name=None, driver=None, driver_info=None, chassis_uuid=None, instance_info=None, instance_uuid=None, properties=None, reservation=None, last_error=None, provision_state='available'):
        self.uuid = id
        self.name = name
        self.driver = driver
        self.driver_info = driver_info
        self.chassis_uuid = chassis_uuid
        self.instance_info = instance_info
        self.instance_uuid = instance_uuid
        self.properties = properties
        self.reservation = reservation
        self.last_error = last_error
        self.provision_state = provision_state