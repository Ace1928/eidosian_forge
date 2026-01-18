from typing import Optional
import requests
class IonQException(Exception):
    """An exception for errors coming from IonQ's API.

    Attributes:
        status_code: A http status code, if coming from an http response with a failing status.
    """

    def __init__(self, message, status_code: Optional[int]=None):
        super().__init__(f"Status code: {status_code}, Message: '{message}'")
        self.status_code = status_code