import collections
import weakref
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.trackable import constants
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.tracking.Trackable', v1=[])
class Trackable(object):
    """Base class for `Trackable` objects without automatic dependencies.

  This class has no __setattr__ override for performance reasons. Dependencies
  must be added explicitly. Unless attribute assignment is performance-critical,
  use `AutoTrackable` instead. Use `Trackable` for `isinstance`
  checks.
  """

    @property
    def _setattr_tracking(self):
        if not hasattr(self, '_self_setattr_tracking'):
            self._self_setattr_tracking = True
        return self._self_setattr_tracking

    @_setattr_tracking.setter
    def _setattr_tracking(self, value):
        self._self_setattr_tracking = value

    @property
    def _update_uid(self):
        return self._self_update_uid

    @_update_uid.setter
    def _update_uid(self, value):
        self._self_update_uid = value

    @property
    def _unconditional_checkpoint_dependencies(self):
        return self._self_unconditional_checkpoint_dependencies

    @property
    def _unconditional_dependency_names(self):
        return self._self_unconditional_dependency_names

    @property
    def _name_based_restores(self):
        return self._self_name_based_restores

    @no_automatic_dependency_tracking
    def _maybe_initialize_trackable(self):
        """Initialize dependency management.

    Not __init__, since most objects will forget to call it.
    """
        if hasattr(self, '_self_unconditional_checkpoint_dependencies'):
            return
        self._self_unconditional_checkpoint_dependencies = []
        self._self_unconditional_dependency_names = {}
        self._self_unconditional_deferred_dependencies = {}
        if hasattr(self, '_self_update_uid'):
            raise AssertionError('Internal error: the object had an update UID set before its initialization code was run.')
        self._self_update_uid = -1
        self._self_name_based_restores = set()
        self._self_saveable_object_factories = {}

    @property
    def _object_identifier(self):
        """String used to identify this object in a SavedModel.

    THIS FIELD HAS BEEN DEPRECATED IN FAVOR OF THE NAME REGISTERED WITH
    `register_serializable`.

    Generally, the object identifier is constant across objects of the same
    class, while the metadata field is used for instance-specific data.

    Returns:
      String object identifier.
    """
        return '_generic_user_object'

    def _no_dependency(self, value):
        """If automatic dependency tracking is enabled, ignores `value`."""
        return value

    def _name_based_attribute_restore(self, checkpoint):
        """Restore the object's attributes from a name-based checkpoint."""
        self._self_name_based_restores.add(checkpoint)
        if self._self_update_uid < checkpoint.restore_uid:
            checkpoint.eager_restore(self)
            self._self_update_uid = checkpoint.restore_uid

    @property
    def _checkpoint_dependencies(self):
        """All dependencies of this object.

    May be overridden to include conditional dependencies.

    Returns:
      A list of `TrackableReference` objects indicating named
      `Trackable` dependencies which should be saved along with this
      object.
    """
        return self._self_unconditional_checkpoint_dependencies

    @property
    def _deferred_dependencies(self):
        """A dictionary with deferred dependencies.

    Stores restorations for other Trackable objects on which this object
    may eventually depend. May be overridden by sub-classes (e.g. Optimizers use
    conditional dependencies based the current graph, and so need separate
    management of deferred dependencies too).

    Returns:
      A dictionary mapping from local name to a list of CheckpointPosition
      objects.
    """
        return self._self_unconditional_deferred_dependencies

    def _lookup_dependency(self, name, cached_dependencies=None):
        """Look up a dependency by name.

    May be overridden to include conditional dependencies.

    Args:
      name: The local name of the dependency.
      cached_dependencies: Optional dict containing all computed dependencies
        returned by `self._trackable_children()`.

    Returns:
      A `Trackable` object, or `None` if no dependency by this name was
      found.
    """
        if cached_dependencies:
            return cached_dependencies.get(name)
        return self._self_unconditional_dependency_names.get(name)

    def _add_variable_with_custom_getter(self, name, shape=None, dtype=dtypes.float32, initializer=None, getter=None, overwrite=False, **kwargs_for_getter):
        """Restore-on-create for a variable be saved with this `Trackable`.

    If the user has requested that this object or another `Trackable` which
    depends on this object be restored from a checkpoint (deferred loading
    before variable object creation), `initializer` may be ignored and the value
    from the checkpoint used instead.

    Args:
      name: A name for the variable. Must be unique within this object.
      shape: The shape of the variable.
      dtype: The data type of the variable.
      initializer: The initializer to use. Ignored if there is a deferred
        restoration stored in the Trackable.
      getter: The getter to wrap which actually fetches the variable.
      overwrite: If True, disables unique name and type checks.
      **kwargs_for_getter: Passed to the getter.

    Returns:
      The new variable object.

    Raises:
      ValueError: If the variable name is not unique.
    """
        self._maybe_initialize_trackable()
        with ops.init_scope():
            if context.executing_eagerly():
                checkpoint_initializer = self._preload_simple_restoration(name=name)
            else:
                checkpoint_initializer = None
            if checkpoint_initializer is not None and (not (isinstance(initializer, CheckpointInitialValueCallable) and initializer.restore_uid > checkpoint_initializer.restore_uid)):
                initializer = checkpoint_initializer
        new_variable = getter(name=name, shape=shape, dtype=dtype, initializer=initializer, **kwargs_for_getter)
        if not overwrite or isinstance(new_variable, Trackable):
            return self._track_trackable(new_variable, name=name, overwrite=overwrite)
        else:
            return new_variable

    def _preload_simple_restoration(self, name):
        """Return a dependency's value for restore-on-create.

    Note the restoration is not deleted; if for some reason preload is called
    and then not assigned to the variable (for example because a custom getter
    overrides the initializer), the assignment will still happen once the
    variable is tracked (determined based on checkpoint.restore_uid).

    Args:
      name: The object-local name of the dependency holding the variable's
        value.

    Returns:
      An callable for use as a variable's initializer/initial_value, or None if
      one should not be set (either because there was no variable with this name
      in the checkpoint or because it needs more complex deserialization). Any
      non-trivial deserialization will happen when the variable object is
      tracked.
    """
        deferred_dependencies_list = self._deferred_dependencies.get(name, ())
        if not deferred_dependencies_list:
            return
        for checkpoint_position in deferred_dependencies_list:
            if not checkpoint_position.is_simple_variable():
                return None
        checkpoint_position = max(deferred_dependencies_list, key=lambda restore: restore.checkpoint.restore_uid)
        return CheckpointInitialValueCallable(checkpoint_position=checkpoint_position)

    def _track_trackable(self, trackable, name, overwrite=False):
        """Declare a dependency on another `Trackable` object.

    Indicates that checkpoints for this object should include variables from
    `trackable`.

    Variables in a checkpoint are mapped to `Trackable`s based on the names
    provided when the checkpoint was written. To avoid breaking existing
    checkpoints when modifying a class, neither variable names nor dependency
    names (the names passed to `_track_trackable`) may change.

    Args:
      trackable: A `Trackable` which this object depends on.
      name: A local name for `trackable`, used for loading checkpoints into the
        correct objects.
      overwrite: Boolean, whether silently replacing dependencies is OK. Used
        for __setattr__, where throwing an error on attribute reassignment would
        be inappropriate.

    Returns:
      `trackable`, for convenience when declaring a dependency and
      assigning to a member variable in one statement.

    Raises:
      TypeError: If `trackable` does not inherit from `Trackable`.
      ValueError: If another object is already tracked by this name.
    """
        self._maybe_initialize_trackable()
        if not isinstance(trackable, Trackable):
            raise TypeError(f'Trackable._track_trackable() can only be used to track objects of type Trackable. Got type {type(trackable)}.')
        if not getattr(self, '_manual_tracking', True):
            return trackable
        new_reference = TrackableReference(name=name, ref=trackable)
        current_object = self._lookup_dependency(name)
        if current_object is not None and current_object is not trackable:
            if not overwrite:
                raise ValueError(f"Called Trackable._track_trackable() with name='{name}', but a Trackable with this name is already declared as a dependency. Names must be unique (or overwrite=True).")
            for index, (old_name, _) in enumerate(self._self_unconditional_checkpoint_dependencies):
                if name == old_name:
                    self._self_unconditional_checkpoint_dependencies[index] = new_reference
        elif current_object is None:
            self._self_unconditional_checkpoint_dependencies.append(new_reference)
            self._handle_deferred_dependencies(name=name, trackable=trackable)
        self._self_unconditional_dependency_names[name] = trackable
        return trackable

    def _handle_deferred_dependencies(self, name, trackable):
        """Pop and load any deferred checkpoint restores into `trackable`.

    This method does not add a new dependency on `trackable`, but it does
    check if any outstanding/deferred dependencies have been queued waiting for
    this dependency to be added (matched based on `name`). If so,
    `trackable` and its dependencies are restored. The restorations are
    considered fulfilled and so are deleted.

    `_track_trackable` is more appropriate for adding a
    normal/unconditional dependency, and includes handling for deferred
    restorations. This method allows objects such as `Optimizer` to use the same
    restoration logic while managing conditional dependencies themselves, by
    overriding `_checkpoint_dependencies` and `_lookup_dependency` to change the
    object's dependencies based on the context it is saved/restored in (a single
    optimizer instance can have state associated with multiple graphs).

    Args:
      name: The name of the dependency within this object (`self`), used to
        match `trackable` with values saved in a checkpoint.
      trackable: The Trackable object to restore (inheriting from `Trackable`).
    """
        self._maybe_initialize_trackable()
        trackable._maybe_initialize_trackable()
        deferred_dependencies_list = self._deferred_dependencies.pop(name, ())
        for checkpoint_position in sorted(deferred_dependencies_list, key=lambda restore: restore.checkpoint.restore_uid, reverse=True):
            checkpoint_position.restore(trackable)
        for name_based_restore in sorted(self._self_name_based_restores, key=lambda checkpoint: checkpoint.restore_uid, reverse=True):
            trackable._name_based_attribute_restore(name_based_restore)

    def _gather_saveables_for_checkpoint(self):
        """Returns a dictionary of values to checkpoint with this object.

    NOTE: This method is deprecated, prefer implementing `_serialize_to_tensors`
    and `_restore_from_tensors` instead. This method is only used in the
    deprecated `tf.compat.v1.train.Saver`.

    Keys in the returned dictionary are local to this object and in a separate
    namespace from dependencies. Values may either be `SaveableObject` factories
    or variables easily converted to `SaveableObject`s (as in
    `tf.compat.v1.train.Saver`'s
    `var_list` constructor argument).

    `SaveableObjects` have a name set, which Trackable needs to generate
    itself. So rather than returning `SaveableObjects` directly, this method
    should return a dictionary of callables which take `name` arguments and
    return `SaveableObjects` with that name.

    If this object may also be passed to the global-name-based
    `tf.compat.v1.train.Saver`,
    the returned callables should have a default value for their name argument
    (i.e. be callable with no arguments).

    Returned values must be saved only by this object; if any value may be
    shared, it should instead be a dependency. For example, variable objects
    save their own values with the key `VARIABLE_VALUE_KEY`, but objects which
    reference variables simply add a dependency.

    Returns:
      The dictionary mapping attribute names to `SaveableObject` factories
      described above. For example:
      {VARIABLE_VALUE_KEY:
       lambda name="global_name_for_this_object":
       SaveableObject(name=name, ...)}
    """
        return getattr(self, '_self_saveable_object_factories', {})

    def _serialize_to_tensors(self):
        """Gathers tensors to save to the checkpoint.

    You should only override `_serialize_to_tensors` and `_restore_from_tensors`
    if you are defining a custom resource or variable with custom ops.

    Otherwise, please store the state of your trackable in `tf.Variable` objects
    and add them to Trackable object hierarchy using `setattr` (for subclasses
    of `AutoTrackable`) or overriding the `_trackable_children` method.

    For an example of a valid implementation of these two methods, please see
    `DenseHashTable`.

    **Invalid implementation**

    ````
    class NamedTrackable(Trackable):
      def __init__(self, name: str):
        self.name = name
      def _serialize_to_tensors(self):
        return {"name": self.name}
      def _restore_from_tensors(self, restored_tensors):
        self.name = restored_tensors["name"]
    ```

    In this example, `NamedTrackable` can be saved and restored from
    checkpoints, but is incompatible with SavedModel, which tries to convert
    the serialize/restore functions into tf.functions. This fails because
    attribute assignment (`self.attr = new_value`) is not graph-friendly.

    **Suggested fix**

    ```
    class NamedTrackable(Trackable):
      def __init__(self, name: str):
        self.name = tf.Variable(name)

      def _trackable_children(self):
        return {"name": self.name}
    ```

    If the `name` attribute should be saved to the checkpoint, then convert it
    a `tf.Variable`.

    **TF1 Saver Compatibility**
    If your Trackable needs to be comatible with `tf.compat.v1.train.Saver`,
    implement `_gather_saveables_from_checkpoint`.

    Returns:
      A dictionary mapping names to tensors.
    """
        raise NotImplementedError

    def _restore_from_tensors(self, restored_tensors):
        """Restores checkpointed values to this `Trackable`.

    Please see the documentation for `Trackable._serialize_to_tensors`.

    Args:
      restored_tensors: A dictionary mapping names to tensors. The keys to this
        dictionary matches the names passed to _serialize_to_tensors.

    Returns:
      An op that runs the restoration.
    """
        raise NotImplementedError

    def _serialize_to_proto(self, object_proto=None, **kwargs):
        """Returns a proto of any type to be saved into the SavedModel.

    Trackable classes decorated with `register_serializable` should overwrite
    this method to save metadata for this object to the SavedModel. The proto
    returned by this function will be passed to `_deserialize_from_proto` in the
    form of a `google.protobuf.Any` proto.

    This data is only saved and used by the Python API. Existing C++ loading
    APIs such as `tensorflow::LoadSavedModel` will not read this field at all.

    Args:
      object_proto: A `SavedObject` proto that may be filled by this function.
        Only the core serializable types (Variable, Function, Constant, Asset)
        should modify this argument.
      **kwargs: Future keyword arguments passed to the object during saving.

    Returns:
      A proto that serializes this class's type.
    """
        del object_proto, kwargs
        return None

    @classmethod
    def _deserialize_from_proto(cls, proto=None, dependencies=None, object_proto=None, export_dir=None, asset_file_def=None, operation_attributes=None, **kwargs):
        """Returns a new object restored by the SavedModel.

    Trackable classes decorated with `register_serializable` should overwrite
    this method to change how the object is loaded from SavedModel. By default,
    the object is initialized with no arguments.

    Example:

    ```
    def _serialize_to_proto(self, **unused_kwargs):
      return Message(name="a")

    @classmethod
    def _deserialize_from_proto(cls, proto, **unused_kwargs):
      if proto.Is(Message.DESCRIPTOR):
        unpacked = Message()
        proto.Unpack(unpacked)
        return cls(unpacked.name)
      else:
        return cls()
    ```

    This function is only used by the Python API. C++ and TensorFlow Serving do
    not have access to your registered class and cannot execute any of the
    non-tf.functions attached to the Python class. However, all signatures and
    tf.functions are still accessible.

    **Avoid creating duplicate trackables**

    SavedModel is saved by recursively gathering all of the trackables and their
    children. SavedModel loading reverses those steps by creating all
    trackables, then reconnecting the children trackables to their parents using
    `Trackable._add_trackable_child`.

    That means that if `_deserialize_from_proto` calls the `__init__` function,
    which creates all of the children trackables, then those children end up
    being created *twice*.

    To avoid this, structure your code so that Trackables are not created
    when deserialized from SavedModel:

    ```
    @register_serializable()
    class Serializable(trackable):
      def __init __(self, from_proto=False):
        create_non_trackable_objects()
        if not from_proto:
          create_variables_and_other_trackables()

      def _deserialize_from_proto(cls, **kwargs):
        return cls(from_proto=True)

      def _add_trackable_child(self, name, value):
        self.__setattr__(name, value)
    ```

    Args:
      proto: A `google.protobuf.Any` proto read from the `SavedModel`.
      dependencies: A dictionary mapping names to dependencies (see
        `_deserialization_dependencies`)
      object_proto: The `SavedObject` proto for this object.
      export_dir: The `SavedModel` directory
      asset_file_def: The `MetaGraphDef`'s `asset_file_def` field.
      operation_attributes: Dictionary mapping nodes to attribute from the
        imported `GraphDef`.
      **kwargs: Future keyword arguments passed to the object when loading.

    Returns:
      A new object.
    """
        del (proto, dependencies, object_proto, export_dir, asset_file_def, operation_attributes, kwargs)
        return cls()

    def _add_trackable_child(self, name, value):
        """Restores a connection between trackables when loading from SavedModel.

    SavedModel stores both the object metadata and its list of children. When
    loading, this function is used along with `_deserialize_from_proto` to load
    objects from the SavedModel: First, all saved objects are created with
    `_deserialize_from_proto`. After that is complete, the children are
    connected using `_add_trackable_child`.

    **Example**

    `tf.Module`, `tf.keras.Model` and Keras layers use `__setattr__` to track
    children. This is why users can call `model.v = tf.Variable(...)`, and the
    variable will be automatically saved to the checkpoint. The implementation
    of this method for the listed objects is:

    ```
    def _add_trackable_child(self, name, value):
      self.__setattr__(name, value)
    ```

    Args:
      name: The name of the connection between the parent and child `Trackable`.
      value: The child `Trackable` object.
    """
        self._track_trackable(value, name, overwrite=True)

    def _deserialization_dependencies(self, children):
        """Returns a dictionary containing `Trackables` that this object depends on.

    Dependencies define the order to serialize and deserialize objects in the
    SavedModel. For example:

    class A(Trackable):
      b = B()
      def _deserialization_dependencies(self, children):
        return {'b': self.b}

    class B(Trackable):
      pass

    We say that object `a=A()` depends on `a.b`.

    Dependencies are guaranteed to be serialized and deserialized before the
    object depending on them. The following methods use dependencies:
      - `_deserialize_from_proto` [loading]

    SavedModel loads with the bottom-up approach, by first creating all objects
    in the order defined by the dependencies, then connecting the children.

    Unlike `_trackable_children`, this function does not define the
    `SavedObjectGraph`. It only changes the order in which things are
    saved/loaded. Therefore, if there are dependencies that are not in the
    `SavedObjectGraph`, saving will fail.

    Args:
      children: Dict returned from `_trackable_children`.

    Returns:
      A dictionary mapping names to `Trackable`.
    """
        del children
        return {}

    def _trackable_children(self, save_type=SaveType.CHECKPOINT, cache=None, **kwargs):
        """Returns this object's `Trackable` attributes.

    This method is used to build the object graph (or the object hierarchy,
    in pickling terms) for checkpoint save/restore, and `SavedModel` export.

    Override this method to define the children of this instance. Please read
    the implementation restrictions:

    **Rule 1: All children must be convertable to `Trackable`.**

    Must pass `isinstance` check or `converter.convert_to_trackable`.

    **Rule 2: [Checkpoint-only] Do not create new objects.**

    When saving to a `SavedModel`, this method is called *exactly once* for each
    `Trackable` in the object graph. When saving or restoring from a checkpoint,
    this method may be called *multiple times*. Thus, this method may create
    new Trackables when `save_type == SaveType.SAVEDMODEL` but not when
    `save_type == SaveType.CHECKPOINT`.

    When saving to `SavedModel`, new `Trackable` children can be created to save
    non-Trackable attributes to the `SavedModel`. In the example below, `hyper`
    is a regular python float hyperparameter. To save this value, a new Variable
    is created to store the value of `hyper`:

    ```
    def __init__(self):
      self.hyper = 1e-5

    def _trackable_children(self, save_type, **unused_kwargs):
      # Correct implementation
      children = {}
      if format == 'saved_model':
        children['hyper'] = tf.Variable(self.hyper)
      return children
    ```

    An incorrect implementation of `_trackable_children` is shown below. This
    function would cause failures when loading the checkpoint, and calling
    `load_status.assert_consumed()` or
    `load_status.assert_existing_objects_matched`. If you want a value to be
    saved in the checkpoint, hyper must be defined as a `tf.Variable` from the
    start.

    ```
    def _trackable_children(self, save_type, **unused_kwargs):
      # Incorrect implementation
      return {'hyper': tf.Variable(self.hyper)}
    ```

    **Rule 3: [`SavedModel`-only] Watch out for un-traced tf.functions.**

    At the begining of `_trackable_children`, always call
    `get_concrete_function()` for any `tf.function` that has an input signature.

    When `tf.functions` are saved to `SavedModel`, any `tf.functions` that have
    an input signature and has never been called is traced at export time in
    order to copy the op graph into the `SavedModel`. `tf.functions` that are
    traced for the first time are allowed to create new state:


    ```
    @tf.function(input_signature=[]):
    def fn(self);
      if self.v is None:
        self.v = tf.Variable(1.)
      return self.v
    ```

    A problem occurs when there is a `Trackable` that returns `fn` as one of its
    children and `self.v` has not been created yet. When `fn` is traced,
    `self.v` is added to the `Trackable`, but `SavedModel` does not see this
    modification since the `Trackable`'s children have already been gathered.

    Therefore, as a precaution, call `get_concrete_function()` at the very
    start of `_trackable_children` to ensure that the function is traced:


    ```
    def _trackable_children(self):
      self.fn.get_concrete_function()
      return {"v": self.v, "fn": self.fn}
    ```

    Args:
      save_type: A string, can be 'savedmodel' or 'checkpoint'. Defaults to
        SaveType.CHECKPOINT.
      cache: May be `None`, or a dictionary. When `save_type == savedmodel`, a
        new cache is created at the start of the SavedModel export, and shared
        between all `Trackables` in the same object graph. This cache may be
        used for advanced saving functionality.
      **kwargs: Additional kwargs that may be added at a later time.

    Returns:
      Dictionary mapping names to child trackables.
    """
        del save_type, cache, kwargs
        self._maybe_initialize_trackable()
        return {name: ref for name, ref in self._checkpoint_dependencies}

    def _export_to_saved_model_graph(self, object_map, tensor_map, options, **kwargs):
        """Creates a copy of this object's tensors onto SavedModel graph.

    Needs to be overridden if the class contains tensors that must be saved
    into the graph. This method should update the `object_map` and `tensor_map`
    dictionaries.

    This method is called on all nodes in the Trackable Graph (generated by
    `_trackable_children`). The nodes are traversed in the order defined by
    `_deserialization_dependencies`

    All usages of _map_resources should be migrated to this method.

    Args:
      object_map: A dictionary that maps original Trackables to the copied
        Trackables. This only needs to be updated if the object is a
        tf.function, or if the copied tensors are necessary for checkpointing
        this object.
      tensor_map: Dictionary mapping original tensors to copied tensors.
      options: A `tf.saved_model.SaveOptions` object.
      **kwargs: Additional kwargs that may be added at a later time.

    Returns:
      Flat list of original tensors that have been copied.
    """
        _, _, _ = (object_map, tensor_map, options)
        del kwargs
        return []