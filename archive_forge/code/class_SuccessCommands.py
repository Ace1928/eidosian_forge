import os.path
import subprocess
import sys
from unittest import mock
from oslo_config import cfg
from oslotest import base
from oslo_upgradecheck import upgradecheck
class SuccessCommands(TestCommands):
    _upgrade_checks = ()