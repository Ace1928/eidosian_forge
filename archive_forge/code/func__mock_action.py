import testtools
from unittest import mock
from troveclient import base
from troveclient.v1 import management
def _mock_action(self):
    self.body_ = ''

    def side_effect_func(instance_id, body):
        self.body_ = body
    self.management._action = mock.Mock(side_effect=side_effect_func)