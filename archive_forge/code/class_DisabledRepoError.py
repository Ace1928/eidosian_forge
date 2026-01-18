import re
from typing import Optional
from requests import HTTPError, Response
from ._fixes import JSONDecodeError
class DisabledRepoError(HfHubHTTPError):
    """
    Raised when trying to access a repository that has been disabled by its author.

    Example:

    ```py
    >>> from huggingface_hub import dataset_info
    >>> dataset_info("laion/laion-art")
    (...)
    huggingface_hub.utils._errors.DisabledRepoError: 403 Client Error. (Request ID: Root=1-659fc3fa-3031673e0f92c71a2260dbe2;bc6f4dfb-b30a-4862-af0a-5cfe827610d8)

    Cannot access repository for url https://huggingface.co/api/datasets/laion/laion-art.
    Access to this resource is disabled.
    ```
    """