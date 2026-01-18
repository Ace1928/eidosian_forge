from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from gae_ext_runtime import ext_runtime
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
class CoreRuntimeLoader(object):
    """A loader stub for the core runtimes.

  The externalized core runtimes are currently distributed with the cloud sdk.
  This class encapsulates the name of a core runtime to avoid having to load
  it at module load time.  Instead, the wrapped runtime is demand-loaded when
  the Fingerprint() method is called.
  """

    def __init__(self, name, visible_name, allowed_runtime_names):
        self._name = name
        self._rep = None
        self._visible_name = visible_name
        self._allowed_runtime_names = allowed_runtime_names

    @property
    def ALLOWED_RUNTIME_NAMES(self):
        return self._allowed_runtime_names

    @property
    def NAME(self):
        return self._visible_name

    def Fingerprint(self, path, params):
        if not self._rep:
            path_to_runtime = os.path.join(_GetRuntimeDefDir(), self._name)
            self._rep = ext_runtime.ExternalizedRuntime.Load(path_to_runtime, GCloudExecutionEnvironment())
        return self._rep.Fingerprint(path, params)