def _CallMethod(self, srvc, method_descriptor, rpc_controller, request, callback):
    """Calls the method described by a given method descriptor.

    Args:
      srvc: Instance of the service for which this method is called.
      method_descriptor: Descriptor that represent the method to call.
      rpc_controller: RPC controller to use for this method's execution.
      request: Request protocol message.
      callback: A callback to invoke after the method has completed.
    """
    if method_descriptor.containing_service != self.descriptor:
        raise RuntimeError('CallMethod() given method descriptor for wrong service type.')
    method = getattr(srvc, method_descriptor.name)
    return method(rpc_controller, request, callback)