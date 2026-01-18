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
class TestContextGreen(GreenTest, TestContext):
    """gevent subclass of context tests"""
    test_gc = GreenTest.skip_green
    test_term_thread = GreenTest.skip_green
    test_destroy_linger = GreenTest.skip_green
    _repr_cls = 'zmq.green.Context'