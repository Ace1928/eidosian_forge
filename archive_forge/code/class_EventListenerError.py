class EventListenerError(Error):
    """Top level exception raised by YAML listener.

  Any exception raised within the process of parsing a YAML file via an
  EventListener is caught and wrapped in an EventListenerError.  The causing
  exception is maintained, but additional useful information is saved which
  can be used for reporting useful information to users.

  Attributes:
    cause: The original exception which caused the EventListenerError.
  """

    def __init__(self, cause):
        """Initialize event-listener error."""
        if hasattr(cause, 'args') and cause.args:
            Error.__init__(self, *cause.args)
        else:
            Error.__init__(self, str(cause))
        self.cause = cause

    def __str__(self):
        return str(self.cause)