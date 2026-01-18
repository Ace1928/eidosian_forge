from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import logging
import os
import subprocess
import sys
import threading
from . import comm
import ruamel.yaml as yaml
from six.moves import input
class PluginResult(object):

    def __init__(self):
        self.exit_code = -1
        self.runtime_data = None
        self.generated_appinfo = None
        self.docker_context = None
        self.files = []