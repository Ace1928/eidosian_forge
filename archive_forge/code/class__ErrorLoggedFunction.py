import traceback, logging
from OpenGL._configflags import ERROR_LOGGING, FULL_LOGGING
class _ErrorLoggedFunction(_LoggedFunction):
    """On-error-logged function wrapper"""

    def __call__(self, *args, **named):
        function = getattr(self, '')
        try:
            return function(*args, **named)
        except Exception as err:
            self.log.warning('Failure on %s: %s', function.__name__, self.log.getException(err))
            raise