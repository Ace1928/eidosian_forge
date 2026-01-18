from __future__ import annotations
import uuid
from collections import defaultdict
from typing import Sequence
from streamlit import util
from streamlit.runtime.stats import CacheStat, group_stats
from streamlit.runtime.uploaded_file_manager import (
class MemoryUploadedFileManager(UploadedFileManager):
    """Holds files uploaded by users of the running Streamlit app.
    This class can be used safely from multiple threads simultaneously.
    """

    def __init__(self, upload_endpoint: str):
        self.file_storage: dict[str, dict[str, UploadedFileRec]] = defaultdict(dict)
        self.endpoint = upload_endpoint

    def get_files(self, session_id: str, file_ids: Sequence[str]) -> list[UploadedFileRec]:
        """Return a  list of UploadedFileRec for a given sequence of file_ids.

        Parameters
        ----------
        session_id
            The ID of the session that owns the files.
        file_ids
            The sequence of ids associated with files to retrieve.

        Returns
        -------
        List[UploadedFileRec]
            A list of URL UploadedFileRec instances, each instance contains information
            about uploaded file.
        """
        session_storage = self.file_storage[session_id]
        file_recs = []
        for file_id in file_ids:
            file_rec = session_storage.get(file_id, None)
            if file_rec is not None:
                file_recs.append(file_rec)
        return file_recs

    def remove_session_files(self, session_id: str) -> None:
        """Remove all files associated with a given session."""
        self.file_storage.pop(session_id, None)

    def __repr__(self) -> str:
        return util.repr_(self)

    def add_file(self, session_id: str, file: UploadedFileRec) -> None:
        """
        Safe to call from any thread.

        Parameters
        ----------
        session_id
            The ID of the session that owns the file.
        file
            The file to add.
        """
        self.file_storage[session_id][file.file_id] = file

    def remove_file(self, session_id, file_id):
        """Remove file with given file_id associated with a given session."""
        session_storage = self.file_storage[session_id]
        session_storage.pop(file_id, None)

    def get_upload_urls(self, session_id: str, file_names: Sequence[str]) -> list[UploadFileUrlInfo]:
        """Return a list of UploadFileUrlInfo for a given sequence of file_names."""
        result = []
        for _ in file_names:
            file_id = str(uuid.uuid4())
            result.append(UploadFileUrlInfo(file_id=file_id, upload_url=f'{self.endpoint}/{session_id}/{file_id}', delete_url=f'{self.endpoint}/{session_id}/{file_id}'))
        return result

    def get_stats(self) -> list[CacheStat]:
        """Return the manager's CacheStats.

        Safe to call from any thread.
        """
        all_files: list[UploadedFileRec] = []
        file_storage_copy = self.file_storage.copy()
        for session_storage in file_storage_copy.values():
            all_files.extend(session_storage.values())
        stats: list[CacheStat] = [CacheStat(category_name='UploadedFileManager', cache_name='', byte_length=len(file.data)) for file in all_files]
        return group_stats(stats)