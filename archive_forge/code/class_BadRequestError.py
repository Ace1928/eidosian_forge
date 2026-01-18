import re
from typing import Optional
from requests import HTTPError, Response
from ._fixes import JSONDecodeError
class BadRequestError(HfHubHTTPError, ValueError):
    """
    Raised by `hf_raise_for_status` when the server returns a HTTP 400 error.

    Example:

    ```py
    >>> resp = requests.post("hf.co/api/check", ...)
    >>> hf_raise_for_status(resp, endpoint_name="check")
    huggingface_hub.utils._errors.BadRequestError: Bad request for check endpoint: {details} (Request ID: XXX)
    ```
    """