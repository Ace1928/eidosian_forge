import http.client as http
import os
import re
import time
import psutil
import requests
from glance.tests import functional
from glance.tests.utils import execute
def _get_parent(self, server):
    if server == 'api':
        return self.api_server.process_pid