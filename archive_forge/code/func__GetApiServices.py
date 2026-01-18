import unittest
import httplib2
import six
from apitools.base.protorpclite import messages
from apitools.base.py import base_api
from apitools.base.py import exceptions
from apitools.base.py.testing import mock
from samples.fusiontables_sample.fusiontables_v1 import \
from samples.fusiontables_sample.fusiontables_v1 import \
def _GetApiServices(api_client_class):
    return dict(((name, potential_service) for name, potential_service in six.iteritems(api_client_class.__dict__) if isinstance(potential_service, type) and issubclass(potential_service, base_api.BaseApiService)))