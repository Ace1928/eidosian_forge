from enum import Enum
from typing import Union, Callable, Optional, cast
class InvalidCredsError(ProviderError):
    """Exception used when invalid credentials are used on a provider."""

    def __init__(self, value='Invalid credentials with the provider', driver=None):
        super().__init__(value, http_code=401, driver=driver)