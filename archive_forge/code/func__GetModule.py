from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from gae_ext_runtime import ext_runtime
from googlecloudsdk.api_lib.app import ext_runtime_adapter
from googlecloudsdk.api_lib.app.runtimes import python
from googlecloudsdk.api_lib.app.runtimes import python_compat
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
def _GetModule(path, params=None, config_filename=None):
    """Helper function for generating configs.

  Args:
    path: (basestring) Root directory to identify.
    params: (ext_runtime.Params or None) Parameters passed through to the
      fingerprinters.  Uses defaults if not provided.
    config_filename: (str or None) Filename of the config file (app.yaml).

  Raises:
    UnidentifiedDirectoryError: No runtime module matched the directory.
    ConflictingConfigError: Current app.yaml conflicts with other params.

  Returns:
    ext_runtime.Configurator, the configurator for the path
  """
    if not params:
        params = ext_runtime.Params()
    config = params.appinfo
    if config and (not params.deploy):
        if not params.custom:
            raise ConflictingConfigError('Configuration file already exists. This command generates an app.yaml configured to run an application on Google App Engine. To create the configuration files needed to run this application with docker, try `gcloud preview app gen-config --custom`.')
        if not config.IsVm():
            raise ConflictingConfigError('gen-config is only supported for App Engine Flexible.  Please use "vm: true" in your app.yaml if you would like to use App Engine Flexible to run your application.')
        if config.GetEffectiveRuntime() != 'custom' and params.runtime is not None and (params.runtime != config.GetEffectiveRuntime()):
            raise ConflictingConfigError('[{0}] contains "runtime: {1}" which conficts with --runtime={2}.'.format(config_filename, config.GetEffectiveRuntime(), params.runtime))
    module = IdentifyDirectory(path, params)
    if not module:
        raise UnidentifiedDirectoryError(path)
    return module