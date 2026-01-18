import copy
import datetime
import random
from unittest import mock
import uuid
import freezegun
import http.client
from oslo_serialization import jsonutils
from pycadf import cadftaxonomy
import urllib
from urllib import parse as urlparse
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import oauth1
from keystone.oauth1.backends import base
from keystone.tests import unit
from keystone.tests.unit.common import test_notifications
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
from keystone.tests.unit import test_v3
def _consumer_create(self, description=None, description_flag=True, **kwargs):
    if description_flag:
        ref = {'description': description}
    else:
        ref = {}
    if kwargs:
        ref.update(kwargs)
    resp = self.post(self.CONSUMER_URL, body={'consumer': ref})
    consumer = resp.result['consumer']
    consumer_id = consumer['id']
    self.assertEqual(description, consumer['description'])
    self.assertIsNotNone(consumer_id)
    self.assertIsNotNone(consumer['secret'])
    return consumer