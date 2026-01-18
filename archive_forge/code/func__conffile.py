import http.client as http
import os
import re
import time
import psutil
import requests
from glance.tests import functional
from glance.tests.utils import execute
def _conffile(self, service):
    conf_dir = os.path.join(self.test_dir, 'etc')
    conf_filepath = os.path.join(conf_dir, '%s.conf' % service)
    return conf_filepath