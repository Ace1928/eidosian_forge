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
class PrepareVolumes(task.Task):

    def execute(self, volume_specs):
        for v in volume_specs:
            with slow_down():
                print('Dusting off your hard drive %s' % v)
            with slow_down():
                print('Taking a well deserved break.')
            print('Your drive %s has been certified.' % v)