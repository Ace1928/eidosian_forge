import contextlib
import hashlib
import logging
import os
import random
import sys
import time
import futurist
from oslo_utils import uuidutils
from taskflow import engines
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.persistence import models
from taskflow import task
import example_utils as eu  # noqa
class BootVM(task.Task):
    """Fires off the vm boot operation."""

    def execute(self, vm_spec):
        print('Starting vm!')
        with slow_down(1):
            print('Created: %s' % vm_spec)