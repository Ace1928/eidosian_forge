import io
import json
import os
import sys
from unittest import mock
import ddt
from osprofiler.cmd import shell
from osprofiler import exc
from osprofiler.tests import test
def _create_mock_notifications(self):
    notifications = {'info': {'started': 0, 'finished': 1, 'name': 'total'}, 'children': [{'info': {'started': 0, 'finished': 1, 'name': 'total'}, 'children': []}]}
    return notifications