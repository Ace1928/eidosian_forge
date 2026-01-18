from __future__ import annotations
import logging
import os
import tempfile
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Sequence, Union
from langchain_core.pydantic_v1 import (
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.blob_loaders.file_system import (
from langchain_community.document_loaders.blob_loaders.schema import Blob
def _auth(self) -> Account:
    """Authenticates the OneDrive API client

        Returns:
            The authenticated Account object.
        """
    try:
        from O365 import Account, FileSystemTokenBackend
    except ImportError:
        raise ImportError('O365 package not found, please install it with `pip install o365`')
    if self.auth_with_token:
        token_storage = _O365TokenStorage()
        token_path = token_storage.token_path
        token_backend = FileSystemTokenBackend(token_path=token_path.parent, token_filename=token_path.name)
        account = Account(credentials=(self.settings.client_id, self.settings.client_secret.get_secret_value()), scopes=self._scopes, token_backend=token_backend, **{'raise_http_errors': False})
    else:
        token_backend = FileSystemTokenBackend(token_path=Path.home() / '.credentials')
        account = Account(credentials=(self.settings.client_id, self.settings.client_secret.get_secret_value()), scopes=self._scopes, token_backend=token_backend, **{'raise_http_errors': False})
        account.authenticate()
    return account