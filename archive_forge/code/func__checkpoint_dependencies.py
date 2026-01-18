from tensorflow.python.util.tf_export import tf_export
@property
def _checkpoint_dependencies(self):
    return self._trackable._checkpoint_dependencies