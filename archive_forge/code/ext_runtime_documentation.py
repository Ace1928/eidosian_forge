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
Do config generation on the runtime.

    This should generally be called from the configurator's GenerateConfigs()
    method.

    Args:
      configurator: (ExternalRuntimeConfigurator) The configurator retuned by
        Detect().

    Returns:
      (bool) True if files were generated, False if not
    