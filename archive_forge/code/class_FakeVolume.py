import datetime
import hashlib
import json
import uuid
from openstack.cloud import meta
from openstack.orchestration.util import template_format
from openstack import utils
class FakeVolume:

    def __init__(self, id, status, name, attachments=[], size=75):
        self.id = id
        self.status = status
        self.name = name
        self.attachments = attachments
        self.size = size
        self.snapshot_id = 'id:snapshot'
        self.description = 'description'
        self.volume_type = 'type:volume'
        self.availability_zone = 'az1'
        self.created_at = '1900-01-01 12:34:56'
        self.updated_at = None
        self.source_volid = '12345'
        self.metadata = {}