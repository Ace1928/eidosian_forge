def BuildService(self, cls):
    """Constructs the service class.

    Args:
      cls: The class that will be constructed.
    """

    def _WrapCallMethod(srvc, method_descriptor, rpc_controller, request, callback):
        return self._CallMethod(srvc, method_descriptor, rpc_controller, request, callback)
    self.cls = cls
    cls.CallMethod = _WrapCallMethod
    cls.GetDescriptor = staticmethod(lambda: self.descriptor)
    cls.GetDescriptor.__doc__ = 'Returns the service descriptor.'
    cls.GetRequestClass = self._GetRequestClass
    cls.GetResponseClass = self._GetResponseClass
    for method in self.descriptor.methods:
        setattr(cls, method.name, self._GenerateNonImplementedMethod(method))