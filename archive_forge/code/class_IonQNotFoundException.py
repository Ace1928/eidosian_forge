from typing import Optional
import requests
class IonQNotFoundException(IonQException):
    """An exception for errors from IonQ's API when a resource is not found."""

    def __init__(self, message):
        super().__init__(message, status_code=requests.codes.not_found)