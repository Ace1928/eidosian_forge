import copy
import logging
import threading
import google.auth.exceptions as e
class RefreshThread(threading.Thread):
    """
    Thread that refreshes credentials.
    """

    def __init__(self, cred, request, **kwargs):
        """Initializes the thread.

        Args:
            cred: A Credential object to refresh.
            request: A Request object used to perform a credential refresh.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self._cred = cred
        self._request = request
        self._error_info = None

    def run(self):
        """
        Perform the credential refresh.
        """
        try:
            self._cred.refresh(self._request)
        except Exception as err:
            _LOGGER.error(f'Background refresh failed due to: {err}')
            self._error_info = err