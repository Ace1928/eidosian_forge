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
def SetGeneratedAppInfo(self, generated_appinfo):
    """Sets the generated appinfo."""
    self.generated_appinfo = generated_appinfo