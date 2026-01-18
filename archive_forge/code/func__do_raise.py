import json
from unittest import mock
import uuid
from openstack import exceptions
from openstack.tests.unit import base
def _do_raise(self, *args, **kwargs):
    return exceptions.raise_from_response(*args, **kwargs)