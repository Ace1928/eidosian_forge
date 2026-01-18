from tensorflow.python.util.tf_export import tf_export
@property
def _unconditional_checkpoint_dependencies(self):
    return self._trackable._unconditional_checkpoint_dependencies