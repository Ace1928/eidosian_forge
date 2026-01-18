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
def Detect(self, path, params):
    """Determine if 'path' contains an instance of the runtime type.

    Checks to see if the 'path' directory looks like an instance of the
    runtime type.

    Args:
      path: (str) The path name.
      params: (Params) Parameters used by the framework.

    Returns:
      (Configurator) An object containing parameters inferred from source
        inspection.
    """
    detect = self.config.get('detect')
    if detect:
        result = self.RunPlugin('detect', detect, params, [path], (0, 1))
        if result.exit_code:
            return None
        else:
            return ExternalRuntimeConfigurator(self, params, result.runtime_data, result.generated_appinfo, path, self.env)
    else:
        return None