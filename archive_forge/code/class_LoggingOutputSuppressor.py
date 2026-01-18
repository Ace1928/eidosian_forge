import logging
class LoggingOutputSuppressor:
    """Context manager to prevent global logger from printing"""

    def __enter__(self):
        self.orig_handlers = logger.handlers
        for handler in self.orig_handlers:
            logger.removeHandler(handler)

    def __exit__(self, exc, value, tb):
        for handler in self.orig_handlers:
            logger.addHandler(handler)