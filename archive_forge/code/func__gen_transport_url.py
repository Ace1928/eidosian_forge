import copy
import logging
import os
import queue
import select
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from unittest import mock
import uuid
from oslo_utils import eventletutils
from oslo_utils import importutils
from string import Template
import testtools
import oslo_messaging
from oslo_messaging.tests import utils as test_utils
def _gen_transport_url(self, hosts):
    url = 'amqp://%s' % ','.join(map(lambda x: '%s:%d' % (x.hostname, x.port), hosts))
    return oslo_messaging.TransportURL.parse(self.conf, url)