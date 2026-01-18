import itertools
import json
import logging
from unittest import mock
from keystoneauth1 import adapter
import requests
from openstack import exceptions
from openstack import format
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
@staticmethod
def _fake_resource(statuses=None, progresses=None, *, attribute='status'):
    if statuses is None:
        statuses = ['building', 'building', 'building', 'active']

    def fetch(*args, **kwargs):
        if statuses:
            setattr(fake_resource, attribute, statuses.pop(0))
        if progresses:
            fake_resource.progress = progresses.pop(0)
        return fake_resource
    spec = ['id', attribute, 'fetch']
    if progresses:
        spec.append('progress')
    fake_resource = mock.Mock(spec=spec)
    setattr(fake_resource, attribute, statuses.pop(0))
    fake_resource.fetch.side_effect = fetch
    return fake_resource