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
def SendResponse(response):
    json.dump(response, plugin_stdin)
    plugin_stdin.write('\n')
    plugin_stdin.flush()