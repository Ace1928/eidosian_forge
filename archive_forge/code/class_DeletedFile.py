from __future__ import annotations
import io
from abc import abstractmethod
from typing import NamedTuple, Protocol, Sequence
from streamlit import util
from streamlit.proto.Common_pb2 import FileURLs as FileURLsProto
from streamlit.runtime.stats import CacheStatsProvider
class DeletedFile(NamedTuple):
    """Represents a deleted file in deserialized values for st.file_uploader and
    st.camera_input

    Return this from st.file_uploader and st.camera_input deserialize (so they can
    be used in session_state), when widget value contains file record that is missing
    from the storage.
    DeleteFile instances filtered out before return final value to the user in script,
    or before sending to frontend."""
    file_id: str