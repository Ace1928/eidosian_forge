import os.path
from tensorboard.compat import tf
def RetrieveAsset(logdir, plugin_name, asset_name):
    """Retrieve a particular plugin asset from a logdir.

    Args:
      logdir: A directory that was created by a TensorFlow summary.FileWriter.
      plugin_name: The plugin we want an asset from.
      asset_name: The name of the requested asset.

    Returns:
      string contents of the plugin asset.

    Raises:
      KeyError: if the asset does not exist.
    """
    asset_path = os.path.join(PluginDirectory(logdir, plugin_name), asset_name)
    try:
        with tf.io.gfile.GFile(asset_path, 'r') as f:
            return f.read()
    except tf.errors.NotFoundError:
        raise KeyError('Asset path %s not found' % asset_path)
    except tf.errors.OpError as e:
        raise KeyError("Couldn't read asset path: %s, OpError %s" % (asset_path, e))