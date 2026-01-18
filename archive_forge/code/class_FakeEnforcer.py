import hashlib
import http.client as http
import os
import subprocess
import tempfile
import time
import urllib
import uuid
import fixtures
from oslo_limit import exception as ol_exc
from oslo_limit import limit
from oslo_serialization import jsonutils
from oslo_utils.secretutils import md5
from oslo_utils import units
import requests
from glance.quota import keystone as ks_quota
from glance.tests import functional
from glance.tests.functional import ft_utils as func_utils
from glance.tests import utils as test_utils
class FakeEnforcer:

    def __init__(self, callback):
        self._callback = callback

    def enforce(self, project_id, values):
        for name, delta in values.items():
            current = self._callback(project_id, values.keys())
            if current.get(name) + delta > limits.get(name, 0):
                raise ol_exc.ProjectOverLimit(project_id=project_id, over_limit_info_list=[ol_exc.OverLimitInfo(name, limits.get(name), current.get(name), delta)])

    def calculate_usage(self, project_id, names):
        return {name: limit.ProjectUsage(limits.get(name, 0), self._callback(project_id, [name])[name]) for name in names}