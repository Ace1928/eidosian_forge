import copy
import gc
import os
import sys
import time
from queue import Queue
from threading import Event, Thread
from unittest import mock
import pytest
from pytest import mark
import zmq
from zmq.tests import PYPY, BaseZMQTestCase, GreenTest, SkipTest
class CyclicReference:

    def __init__(self, parent=None):
        self.parent = parent

    def crash(self, sock):
        self.sock = sock
        self.child = CyclicReference(self)