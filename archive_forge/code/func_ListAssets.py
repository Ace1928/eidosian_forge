import os.path
from tensorboard.compat import tf
def ListAssets(logdir, plugin_name):
    """List all the assets that are available for given plugin in a logdir.

    Args:
      logdir: A directory that was created by a TensorFlow summary.FileWriter.
      plugin_name: A string name of a plugin to list assets for.

    Returns:
      A string list of available plugin assets. If the plugin subdirectory does
      not exist (either because the logdir doesn't exist, or because the plugin
      didn't register) an empty list is returned.
    """
    plugin_dir = PluginDirectory(logdir, plugin_name)
    try:
        return [x.rstrip('/') for x in tf.io.gfile.listdir(plugin_dir)]
    except tf.errors.NotFoundError:
        return []