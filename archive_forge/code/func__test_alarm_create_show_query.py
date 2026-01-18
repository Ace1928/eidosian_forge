import os
from oslo_utils import uuidutils
import requests
import requests.auth
from tempest.lib import exceptions
from aodhclient.tests.functional import base
def _test_alarm_create_show_query(self, create_params, expected_lines):

    def test(params):
        result = self.aodh('alarm', params=params)
        alarm = self.details_multiple(result)[0]
        for key, value in expected_lines.items():
            self.assertEqual(value, alarm[key])
        return alarm
    alarm = test(create_params)
    params = 'show %s' % alarm['alarm_id']
    test(params)
    self.aodh('alarm', params='delete %s' % alarm['alarm_id'])