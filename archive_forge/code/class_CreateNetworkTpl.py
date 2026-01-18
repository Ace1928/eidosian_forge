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
class CreateNetworkTpl(task.Task):
    """Generates the network settings file to be placed in the images."""
    SYSCONFIG_CONTENTS = 'DEVICE=eth%s\nBOOTPROTO=static\nIPADDR=%s\nONBOOT=yes'

    def __init__(self, name):
        super(CreateNetworkTpl, self).__init__(provides='network_settings', name=name)

    def execute(self, ips):
        settings = []
        for i, ip in enumerate(ips):
            settings.append(self.SYSCONFIG_CONTENTS % (i, ip))
        return settings