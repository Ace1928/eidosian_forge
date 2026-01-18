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
class WriteNetworkSettings(task.Task):
    """Writes all the network settings into the downloaded images."""

    def execute(self, download_paths, network_settings):
        for j, path in enumerate(download_paths):
            with slow_down(1):
                print('Mounting %s to /tmp/%s' % (path, j))
            for i, setting in enumerate(network_settings):
                filename = '/tmp/etc/sysconfig/network-scripts/ifcfg-eth%s' % i
                with slow_down(1):
                    print('Writing to %s' % filename)
                    print(setting)