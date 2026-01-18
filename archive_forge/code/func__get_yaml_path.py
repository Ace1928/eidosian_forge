import logging
import os
def _get_yaml_path(builtin_name, runtime):
    """Return expected path to a builtin handler's yaml file without error check.
  """
    runtime_specific = os.path.join(_handler_dir, builtin_name, INCLUDE_FILENAME_TEMPLATE % runtime)
    if runtime and os.path.exists(runtime_specific):
        return runtime_specific
    return os.path.join(_handler_dir, builtin_name, DEFAULT_INCLUDE_FILENAME)