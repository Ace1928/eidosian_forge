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
def _load_from_folder(self, folder: Folder) -> Iterable[Blob]:
    """Lazily load all files from a specified folder of the configured MIME type.

        Args:
            folder: The Folder instance from which the files are to be loaded. This
                Folder instance should represent a directory in a file system where the
                files are stored.

        Yields:
            An iterator that yields Blob instances, which are binary representations of
                the files loaded from the folder.
        """
    file_mime_types = self._fetch_mime_types
    items = folder.get_items()
    with tempfile.TemporaryDirectory() as temp_dir:
        os.makedirs(os.path.dirname(temp_dir), exist_ok=True)
        for file in items:
            if file.is_file:
                if file.mime_type in list(file_mime_types.values()):
                    file.download(to_path=temp_dir, chunk_size=self.chunk_size)
        loader = FileSystemBlobLoader(path=temp_dir)
        yield from loader.yield_blobs()
    if self.recursive:
        for subfolder in folder.get_child_folders():
            yield from self._load_from_folder(subfolder)