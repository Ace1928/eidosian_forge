import contextlib
import functools
import threading
import time
from unittest import mock
import eventlet
from eventlet.green import threading as green_threading
import testscenarios
import futurist
from futurist import periodics
from futurist.tests import base
@periodics.periodic(0.1, run_immediately=True)
def activated_periodic():
    ran_at.append(time.time())