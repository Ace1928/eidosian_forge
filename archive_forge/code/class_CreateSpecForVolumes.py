import contextlib
import hashlib
import logging
import os
import random
import sys
import time
from oslo_utils import uuidutils
from taskflow import engines
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.persistence import models
from taskflow import task
import example_utils  # noqa
class CreateSpecForVolumes(task.Task):

    def execute(self):
        volumes = []
        for i in range(0, random.randint(1, 10)):
            volumes.append({'type': 'disk', 'location': '/dev/vda%s' % (i + 1)})
        return volumes