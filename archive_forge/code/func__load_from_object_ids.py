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
def _load_from_object_ids(self, drive: Drive, object_ids: List[str]) -> Iterable[Blob]:
    """Lazily load files specified by their object_ids from a drive.

        Load files into the system as binary large objects (Blobs) and return Iterable.

        Args:
            drive: The Drive instance from which the files are to be loaded. This Drive
                instance should represent a cloud storage service or similar storage
                system where the files are stored.
            object_ids: A list of object_id strings. Each object_id represents a unique
                identifier for a file in the drive.

        Yields:
            An iterator that yields Blob instances, which are binary representations of
            the files loaded from the drive using the specified object_ids.
        """
    file_mime_types = self._fetch_mime_types
    with tempfile.TemporaryDirectory() as temp_dir:
        for object_id in object_ids:
            file = drive.get_item(object_id)
            if not file:
                logging.warning(f"There isn't a file withobject_id {object_id} in drive {drive}.")
                continue
            if file.is_file:
                if file.mime_type in list(file_mime_types.values()):
                    file.download(to_path=temp_dir, chunk_size=self.chunk_size)
        loader = FileSystemBlobLoader(path=temp_dir)
        yield from loader.yield_blobs()