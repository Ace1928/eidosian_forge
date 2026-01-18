def _GetRequestClass(self, method_descriptor):
    """Returns the class of the request protocol message.

    Args:
      method_descriptor: Descriptor of the method for which to return the
        request protocol message class.

    Returns:
      A class that represents the input protocol message of the specified
      method.
    """
    if method_descriptor.containing_service != self.descriptor:
        raise RuntimeError('GetRequestClass() given method descriptor for wrong service type.')
    return method_descriptor.input_type._concrete_class