import os
from typing import Optional
class InvalidPathError(Exception):
    """InvalidPathError class."""

    def __init__(self, path: str, message: Optional[str]=None):
        self.path = path
        self.message = message or f'{path} does not exist'
        super().__init__(self.message)