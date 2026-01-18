import datetime
import hashlib
import json
import uuid
from openstack.cloud import meta
from openstack.orchestration.util import template_format
from openstack import utils
class FakeZone:

    def __init__(self, id, name, type_, email, description, ttl, masters):
        self.id = id
        self.name = name
        self.type_ = type_
        self.email = email
        self.description = description
        self.ttl = ttl
        self.masters = masters