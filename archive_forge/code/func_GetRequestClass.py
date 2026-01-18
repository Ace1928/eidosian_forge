def GetRequestClass(self, method_descriptor):
    """Returns the class of the request message for the specified method.

    CallMethod() requires that the request is of a particular subclass of
    Message. GetRequestClass() gets the default instance of this required
    type.

    Example:
      method = service.GetDescriptor().FindMethodByName("Foo")
      request = stub.GetRequestClass(method)()
      request.ParseFromString(input)
      service.CallMethod(method, request, callback)
    """
    raise NotImplementedError