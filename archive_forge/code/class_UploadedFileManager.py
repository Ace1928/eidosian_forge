from __future__ import annotations
import io
from abc import abstractmethod
from typing import NamedTuple, Protocol, Sequence
from streamlit import util
from streamlit.proto.Common_pb2 import FileURLs as FileURLsProto
from streamlit.runtime.stats import CacheStatsProvider
class UploadedFileManager(CacheStatsProvider, Protocol):
    """UploadedFileManager protocol, that should be implemented by the concrete
    uploaded file managers.

    It is responsible for:
        - retrieving files by session_id and file_id for st.file_uploader and
            st.camera_input
        - cleaning up uploaded files associated with session on session end

    It should be created during Runtime initialization.

    Optionally UploadedFileManager could be responsible for issuing URLs which will be
    used by frontend to upload files to.
    """

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def remove_session_files(self, session_id: str) -> None:
        """Remove all files associated with a given session."""
        raise NotImplementedError

    def get_upload_urls(self, session_id: str, file_names: Sequence[str]) -> list[UploadFileUrlInfo]:
        """Return a list of UploadFileUrlInfo for a given sequence of file_names.
        Optional to implement, issuing of URLs could be done by other service.

        Parameters
        ----------
        session_id
            The ID of the session that request URLs.
        file_names
            The sequence of file names for which URLs are requested

        Returns
        -------
        List[UploadFileUrlInfo]
            A list of UploadFileUrlInfo instances, each instance contains information
            about uploaded file URLs.
        """
        raise NotImplementedError