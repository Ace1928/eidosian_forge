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
def RunPlugin(self, section_name, plugin_spec, params, args=None, valid_exit_codes=(0,), runtime_data=None):
    """Run a plugin.

    Args:
      section_name: (str) Name of the config section that the plugin spec is
        from.
      plugin_spec: ({str: str, ...}) A dictionary mapping plugin locales to
        script names
      params: (Params or None) Parameters for the plugin.
      args: ([str, ...] or None) Command line arguments for the plugin.
      valid_exit_codes: (int, ...) Exit codes that will be accepted without
        raising an exception.
      runtime_data: ({str: object, ...}) A dictionary of runtime data passed
        back from detect.

    Returns:
      (PluginResult) A bundle of the exit code and data produced by the plugin.

    Raises:
      PluginInvocationFailed: The plugin terminated with a non-zero exit code.
    """
    if 'python' in plugin_spec:
        normalized_path = _NormalizePath(self.root, plugin_spec['python'])
        result = PluginResult()
        p = subprocess.Popen([self.env.GetPythonExecutable(), normalized_path] + (args if args else []), stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        stderr_thread = threading.Thread(target=self._ProcessPluginStderr, args=(section_name, p.stderr))
        stderr_thread.start()
        stdout_thread = threading.Thread(target=self._ProcessPluginPipes, args=(section_name, p, result, params, runtime_data))
        stdout_thread.start()
        stderr_thread.join()
        stdout_thread.join()
        exit_code = p.wait()
        result.exit_code = exit_code
        if exit_code not in valid_exit_codes:
            raise PluginInvocationFailed('Failed during execution of plugin %s for section %s of runtime %s. rc = %s' % (normalized_path, section_name, self.config.get('name', 'unknown'), exit_code))
        return result
    else:
        logging.error('No usable plugin type found for %s' % section_name)