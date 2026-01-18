import os
import requests
import subprocess
import time
import uuid
import concurrent.futures
from oslo_config import cfg
from testtools import matchers
import oslo_messaging
from oslo_messaging.tests.functional import utils
class _endpoint(object):

    def target(self, ctxt, echo):
        return echo