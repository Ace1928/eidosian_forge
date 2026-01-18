import os.path
import subprocess
import sys
from unittest import mock
from oslo_config import cfg
from oslotest import base
from oslo_upgradecheck import upgradecheck
class TestUpgradeCommands(base.BaseTestCase):

    def test_get_details(self):
        result = upgradecheck.Result(upgradecheck.Code.SUCCESS, '*' * 70)
        upgrade_commands = upgradecheck.UpgradeCommands()
        details = upgrade_commands._get_details(result)
        wrapped = '*' * 60 + '\n  ' + '*' * 10
        self.assertEqual(wrapped, details)

    def test_check(self):
        inst = TestCommands()
        result = inst.check()
        self.assertEqual(upgradecheck.Code.FAILURE, result)