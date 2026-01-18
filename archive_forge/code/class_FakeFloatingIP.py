import datetime
import hashlib
import json
import uuid
from openstack.cloud import meta
from openstack.orchestration.util import template_format
from openstack import utils
class FakeFloatingIP:

    def __init__(self, id, pool, ip, fixed_ip, instance_id):
        self.id = id
        self.pool = pool
        self.ip = ip
        self.fixed_ip = fixed_ip
        self.instance_id = instance_id