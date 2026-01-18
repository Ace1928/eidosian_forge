import logging
class _TraceLoggerAdapter(logging.LoggerAdapter):

    def trace(self, msg, *args, **kwargs):
        """Delegate a trace call to the underlying logger."""
        self.log(TRACE, msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        """Delegate a warning call to the underlying logger."""
        self.warning(msg, *args, **kwargs)