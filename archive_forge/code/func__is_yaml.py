import operator
from unittest import mock
import warnings
from oslo_config import cfg
import stevedore
import testtools
import yaml
from oslo_policy import generator
from oslo_policy import policy
from oslo_policy.tests import base
from oslo_serialization import jsonutils
def _is_yaml(self, data):
    is_yaml = False
    try:
        jsonutils.loads(data)
    except ValueError:
        try:
            yaml.safe_load(data)
            is_yaml = True
        except yaml.scanner.ScannerError:
            pass
    return is_yaml