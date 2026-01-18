import abc
import sys
from typing import Any, Dict, Optional, Union
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder  # type: ignore
class BackendResponse(Protocol):

    @abc.abstractmethod
    def __init__(self, response: requests.Response) -> None:
        ...