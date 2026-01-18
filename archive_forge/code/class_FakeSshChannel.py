import errno
import io
import logging
import multiprocessing
import os
import pickle
import resource
import socket
import stat
import subprocess
import sys
import tempfile
import time
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_concurrency import processutils
class FakeSshChannel(object):

    def __init__(self, rc):
        self.rc = rc

    def recv_exit_status(self):
        return self.rc