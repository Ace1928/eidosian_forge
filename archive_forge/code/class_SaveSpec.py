class SaveSpec:
    """Class used to describe tensor slices that need to be saved."""

    def __init__(self, tensor, slice_spec, name, dtype=None, device=None):
        """Creates a `SaveSpec` object.

    Args:
      tensor: the tensor to save or callable that produces a tensor to save.
        If the value is `None`, the `SaveSpec` is ignored.
      slice_spec: the slice to be saved. See `Variable.SaveSliceInfo`.
      name: the name to save the tensor under.
      dtype: The data type of the Tensor. Required if `tensor` is callable.
        Used for error checking in the restore op.
      device: The device generating and consuming this tensor. Required if
        `tensor` is callable. Used to group objects to save by device.
    """
        self._tensor = tensor
        self.slice_spec = slice_spec
        self.name = name
        if callable(self._tensor):
            if dtype is None or device is None:
                raise AssertionError('When passing a callable `tensor` to a SaveSpec, an explicit dtype and device must be provided.')
            self.dtype = dtype
            self.device = device
        else:
            self.dtype = tensor.dtype
            if device is not None:
                self.device = device
            else:
                self.device = tensor.device

    @property
    def tensor(self):
        return self._tensor() if callable(self._tensor) else self._tensor