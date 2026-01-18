class DispatcherKeyError(KeyError, DispatcherError):
    """Error raised when unknown (sender,signal) set specified"""