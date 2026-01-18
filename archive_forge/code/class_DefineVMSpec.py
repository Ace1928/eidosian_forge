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
class DefineVMSpec(task.Task):
    """Defines a vm specification to be."""

    def __init__(self, name):
        super(DefineVMSpec, self).__init__(provides='vm_spec', name=name)

    def execute(self):
        return {'type': 'kvm', 'disks': 2, 'vcpu': 1, 'ips': 1, 'volumes': 3}