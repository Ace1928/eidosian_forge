import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_community.document_loaders.base import BaseLoader
def _load_documents_from_folder(self, folder_path: str) -> List[Document]:
    """Load documents from a Dropbox folder."""
    dbx = self._create_dropbox_client()
    try:
        from dropbox import exceptions
        from dropbox.files import FileMetadata
    except ImportError:
        raise ImportError('You must run `pip install dropbox')
    try:
        results = dbx.files_list_folder(folder_path, recursive=self.recursive)
    except exceptions.ApiError as ex:
        raise ValueError(f'Could not list files in the folder: {folder_path}. Please verify the folder path and try again.') from ex
    files = [entry for entry in results.entries if isinstance(entry, FileMetadata)]
    documents = [doc for doc in (self._load_file_from_path(file.path_display) for file in files) if doc is not None]
    return documents