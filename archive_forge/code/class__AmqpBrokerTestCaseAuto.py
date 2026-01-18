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
class _AmqpBrokerTestCaseAuto(_AmqpBrokerTestCase):
    """Like _AmqpBrokerTestCase, but starts the broker"""

    @testtools.skipUnless(pyngus, 'proton modules not present')
    def setUp(self):
        super(_AmqpBrokerTestCaseAuto, self).setUp()
        self._broker.start()