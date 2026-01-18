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
def _on_active(link):
    if self._broker._addresser._is_multicast(link.source_address):
        self._blocked_links.add(link)
    else:
        link.add_capacity(10)
        for li in self._blocked_links:
            li.add_capacity(10)