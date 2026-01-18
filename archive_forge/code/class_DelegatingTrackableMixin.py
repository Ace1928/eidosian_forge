from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.tracking.DelegatingTrackableMixin', v1=[])
class DelegatingTrackableMixin(object):
    """A mixin that delegates all Trackable methods to another trackable object.

  DO NOT USE THIS UNLESS YOU ARE THE KERAS LOSS SCALE OPTIMIZER.

  This class must be used with multiple inheritance. A class that subclasses
  Trackable can also subclass this class, which causes all Trackable methods to
  be delegated to the trackable object passed in the constructor.

  A subclass can use this mixin to appear as if it were the trackable passed to
  the constructor, from a Checkpoint's perspective. LossScaleOptimizer uses this
  mixin, so that the checkpoint format for a LossScaleOptimizer is identical to
  the checkpoint format for a normal optimizer. This allows a model to be saved
  with a normal Optimizer and restored with a LossScaleOptimizer, or vice versa.
  The only difference in checkpoint format is that the loss scale is also saved
  with a LossScaleOptimizer.
  """

    def __init__(self, trackable_obj):
        self._trackable = trackable_obj

    @property
    def _setattr_tracking(self):
        return self._trackable._setattr_tracking

    @_setattr_tracking.setter
    def _setattr_tracking(self, value):
        self._trackable._setattr_tracking = value

    @property
    def _update_uid(self):
        return self._trackable._update_uid

    @_update_uid.setter
    def _update_uid(self, value):
        self._trackable._update_uid = value

    @property
    def _unconditional_checkpoint_dependencies(self):
        return self._trackable._unconditional_checkpoint_dependencies

    @property
    def _unconditional_dependency_names(self):
        return self._trackable._unconditional_dependency_names

    @property
    def _name_based_restores(self):
        return self._trackable._name_based_restores

    def _maybe_initialize_trackable(self):
        return self._trackable._maybe_initialize_trackable()

    @property
    def _object_identifier(self):
        return self._trackable._object_identifier

    @property
    def _tracking_metadata(self):
        return self._trackable._tracking_metadata

    def _no_dependency(self, *args, **kwargs):
        return self._trackable._no_dependency(*args, **kwargs)

    def _name_based_attribute_restore(self, *args, **kwargs):
        return self._trackable._name_based_attribute_restore(*args, **kwargs)

    @property
    def _checkpoint_dependencies(self):
        return self._trackable._checkpoint_dependencies

    @property
    def _deferred_dependencies(self):
        return self._trackable._deferred_dependencies

    def _lookup_dependency(self, *args, **kwargs):
        return self._trackable._lookup_dependency(*args, **kwargs)

    def _add_variable_with_custom_getter(self, *args, **kwargs):
        return self._trackable._add_variable_with_custom_getter(*args, **kwargs)

    def _preload_simple_restoration(self, *args, **kwargs):
        return self._trackable._preload_simple_restoration(*args, **kwargs)

    def _track_trackable(self, *args, **kwargs):
        return self._trackable._track_trackable(*args, **kwargs)

    def _handle_deferred_dependencies(self, name, trackable):
        return self._trackable._handle_deferred_dependencies(name, trackable)

    def _gather_saveables_for_checkpoint(self, *args, **kwargs):
        return self._trackable._gather_saveables_for_checkpoint(*args, **kwargs)

    def _trackable_children(self, *args, **kwargs):
        return self._trackable._trackable_children(*args, **kwargs)

    def _deserialization_dependencies(self, *args, **kwargs):
        return self._trackable._deserialization_dependencies(*args, **kwargs)

    def _export_to_saved_model_graph(self, *args, **kwargs):
        return self._trackable._export_to_saved_model_graph(*args, **kwargs)