from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from gae_ext_runtime import ext_runtime
from googlecloudsdk.api_lib.app import ext_runtime_adapter
from googlecloudsdk.api_lib.app.runtimes import python
from googlecloudsdk.api_lib.app.runtimes import python_compat
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
def IdentifyDirectory(path, params=None):
    """Try to identify the given directory.

  As a side-effect, if there is a config file in 'params' with a runtime of
  'custom', this sets params.custom to True.

  Args:
    path: (basestring) Root directory to identify.
    params: (ext_runtime.Params or None) Parameters passed through to the
      fingerprinters.  Uses defaults if not provided.

  Returns:
    (ext_runtime.Configurator or None) Returns a module if we've identified
    it, None if not.
  """
    if not params:
        params = ext_runtime.Params()
    if params.runtime:
        specified_runtime = params.runtime
    elif params.appinfo:
        specified_runtime = params.appinfo.GetEffectiveRuntime()
    else:
        specified_runtime = None
    if specified_runtime == 'custom':
        params.custom = True
    for runtime in RUNTIMES:
        if specified_runtime and runtime.ALLOWED_RUNTIME_NAMES and (specified_runtime not in runtime.ALLOWED_RUNTIME_NAMES):
            log.info('Not checking for [%s] because runtime is [%s]' % (runtime.NAME, specified_runtime))
            continue
        try:
            configurator = runtime.Fingerprint(path, params)
        except ext_runtime.Error as ex:
            raise ExtRuntimeError(ex.message)
        if configurator:
            return configurator
    return None