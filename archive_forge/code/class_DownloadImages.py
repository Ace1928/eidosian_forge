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
class DownloadImages(task.Task):
    """Downloads all the vm images."""

    def __init__(self, name):
        super(DownloadImages, self).__init__(provides='download_paths', name=name)

    def execute(self, image_locations):
        for src, loc in image_locations.items():
            with slow_down(1):
                print('Downloading from %s => %s' % (src, loc))
        return sorted(image_locations.values())