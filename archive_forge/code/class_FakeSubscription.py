from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import mistral as mistral_client_plugin
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
from oslo_serialization import jsonutils
class FakeSubscription(object):

    def __init__(self, queue_name, id=None, ttl=None, subscriber=None, options=None, auto_create=True):
        self.id = id
        self.queue_name = queue_name
        self.ttl = ttl
        self.subscriber = subscriber
        self.options = options

    def update(self, prop_diff):
        allowed_keys = {'subscriber', 'ttl', 'options'}
        for key in prop_diff.keys():
            if key not in allowed_keys:
                raise KeyError(key)

    def delete(self):
        self._deleted = True