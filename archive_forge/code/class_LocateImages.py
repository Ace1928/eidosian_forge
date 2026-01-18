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
class LocateImages(task.Task):
    """Locates where the vm images are."""

    def __init__(self, name):
        super(LocateImages, self).__init__(provides='image_locations', name=name)

    def execute(self, vm_spec):
        image_locations = {}
        for i in range(0, vm_spec['disks']):
            url = 'http://www.yahoo.com/images/%s' % i
            image_locations[url] = '/tmp/%s.img' % i
        return image_locations