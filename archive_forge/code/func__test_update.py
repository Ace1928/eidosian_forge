import copy
import datetime
import os
import tempfile
from unittest import mock
from oslo_serialization import jsonutils
import yaml
from mistralclient.api.v2 import environments
from mistralclient.commands.v2 import environments as environment_cmd
from mistralclient.tests.unit import base
def _test_update(self, content):
    self.client.environments.update.return_value = ENVIRONMENT
    with tempfile.NamedTemporaryFile() as f:
        f.write(content.encode('utf-8'))
        f.flush()
        file_path = os.path.abspath(f.name)
        result = self.call(environment_cmd.Update, app_args=[file_path])
        self.assertEqual(EXPECTED_RESULT, result[1])