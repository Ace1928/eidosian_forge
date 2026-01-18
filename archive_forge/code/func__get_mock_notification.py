from unittest import mock
from heat.common import exception as heat_exception
from heat.engine.clients.os import monasca as client_plugin
from heat.tests import common
from heat.tests import utils
def _get_mock_notification(self):
    notification = dict()
    notification['id'] = self.sample_uuid
    notification['name'] = self.sample_name
    return notification