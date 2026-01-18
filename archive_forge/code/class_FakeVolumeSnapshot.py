import datetime
import hashlib
import json
import uuid
from openstack.cloud import meta
from openstack.orchestration.util import template_format
from openstack import utils
class FakeVolumeSnapshot:

    def __init__(self, id, status, name, description, size=75):
        self.id = id
        self.status = status
        self.name = name
        self.description = description
        self.size = size
        self.created_at = '1900-01-01 12:34:56'
        self.updated_at = None
        self.volume_id = '12345'
        self.metadata = {}
        self.is_forced = False