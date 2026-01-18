import itertools
import os
import pprint
import select
import socket
import threading
import time
import fixtures
from keystoneauth1 import exceptions
import prometheus_client
from requests import exceptions as rexceptions
import testtools.content
from openstack.tests.unit import base
def _add_prometheus_samples(self, exc_info):
    samples = []
    for metric in self._registry.collect():
        for s in metric.samples:
            samples.append(s)
    self.addDetail('prometheus_samples', testtools.content.text_content(pprint.pformat(samples)))