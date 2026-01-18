import contextlib
import os
from unittest import mock
import testtools
from troveclient.apiclient import exceptions
from troveclient import base
from troveclient import common
from troveclient import utils
def _get_mock(self):
    manager = base.Manager()
    manager.api = mock.Mock()
    manager.api.client = mock.Mock()

    def side_effect_func(self, body, loaded=True):
        return body
    manager.resource_class = mock.Mock(side_effect=side_effect_func)
    return manager