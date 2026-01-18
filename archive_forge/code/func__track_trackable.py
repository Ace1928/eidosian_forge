from tensorflow.python.util.tf_export import tf_export
def _track_trackable(self, *args, **kwargs):
    return self._trackable._track_trackable(*args, **kwargs)