import logging
import sys
def _log_exception(self, event):
    """ A handler that logs the exception with the given event.

        Parameters
        ----------
        event : object
            An event object emitted by the notification.
        """
    _logger.exception('Exception occurred in traits notification handler for event object: %r', event)