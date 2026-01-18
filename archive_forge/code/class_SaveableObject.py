class SaveableObject:
    """Base class for saving and restoring saveable objects."""

    def __init__(self, op, specs, name):
        """Creates a `SaveableObject` object.

    Args:
      op: the "producer" object that this class wraps; it produces a list of
        tensors to save.  E.g., a "Variable" object saving its backing tensor.
      specs: a list of SaveSpec, each element of which describes one tensor to
        save under this object. All Tensors must be on the same device.
      name: the name to save the object under.
    """
        self.op = op
        self.specs = specs
        self.name = name

    @property
    def device(self):
        """The device for SaveSpec Tensors."""
        return self.specs[0].device

    def restore(self, restored_tensors, restored_shapes):
        """Restores this object from 'restored_tensors'.

    Args:
      restored_tensors: the tensors that were loaded from a checkpoint
      restored_shapes: the shapes this object should conform to after
        restore, or None.

    Returns:
      An operation that restores the state of the object.

    Raises:
      ValueError: If the object cannot be restored using the provided
        parameters.
    """
        raise ValueError('Calling an abstract method.')