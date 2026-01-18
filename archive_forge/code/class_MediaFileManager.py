from __future__ import annotations
import collections
import threading
from typing import Final
from streamlit.logger import get_logger
from streamlit.runtime.media_file_storage import MediaFileKind, MediaFileStorage
class MediaFileManager:
    """In-memory file manager for MediaFile objects.

    This keeps track of:
    - Which files exist, and what their IDs are. This is important so we can
      serve files by ID -- that's the whole point of this class!
    - Which files are being used by which AppSession (by ID). This is
      important so we can remove files from memory when no more sessions need
      them.
    - The exact location in the app where each file is being used (i.e. the
      file's "coordinates"). This is is important so we can mark a file as "not
      being used by a certain session" if it gets replaced by another file at
      the same coordinates. For example, when doing an animation where the same
      image is constantly replace with new frames. (This doesn't solve the case
      where the file's coordinates keep changing for some reason, though! e.g.
      if new elements keep being prepended to the app. Unlikely to happen, but
      we should address it at some point.)
    """

    def __init__(self, storage: MediaFileStorage):
        self._storage = storage
        self._file_metadata: dict[str, MediaFileMetadata] = dict()
        self._files_by_session_and_coord: dict[str, dict[str, str]] = collections.defaultdict(dict)
        self._lock = threading.Lock()

    def _get_inactive_file_ids(self) -> set[str]:
        """Compute the set of files that are stored in the manager, but are
        not referenced by any active session. These are files that can be
        safely deleted.

        Thread safety: callers must hold `self._lock`.
        """
        file_ids = set(self._file_metadata.keys())
        for session_file_ids_by_coord in self._files_by_session_and_coord.values():
            file_ids.difference_update(session_file_ids_by_coord.values())
        return file_ids

    def remove_orphaned_files(self) -> None:
        """Remove all files that are no longer referenced by any active session.

        Safe to call from any thread.
        """
        _LOGGER.debug('Removing orphaned files...')
        with self._lock:
            for file_id in self._get_inactive_file_ids():
                file = self._file_metadata[file_id]
                if file.kind == MediaFileKind.MEDIA:
                    self._delete_file(file_id)
                elif file.kind == MediaFileKind.DOWNLOADABLE:
                    if file.is_marked_for_delete:
                        self._delete_file(file_id)
                    else:
                        file.mark_for_delete()

    def _delete_file(self, file_id: str) -> None:
        """Delete the given file from storage, and remove its metadata from
        self._files_by_id.

        Thread safety: callers must hold `self._lock`.
        """
        _LOGGER.debug('Deleting File: %s', file_id)
        self._storage.delete_file(file_id)
        del self._file_metadata[file_id]

    def clear_session_refs(self, session_id: str | None=None) -> None:
        """Remove the given session's file references.

        (This does not remove any files from the manager - you must call
        `remove_orphaned_files` for that.)

        Should be called whenever ScriptRunner completes and when a session ends.

        Safe to call from any thread.
        """
        if session_id is None:
            session_id = _get_session_id()
        _LOGGER.debug('Disconnecting files for session with ID %s', session_id)
        with self._lock:
            if session_id in self._files_by_session_and_coord:
                del self._files_by_session_and_coord[session_id]
        _LOGGER.debug('Sessions still active: %r', self._files_by_session_and_coord.keys())
        _LOGGER.debug('Files: %s; Sessions with files: %s', len(self._file_metadata), len(self._files_by_session_and_coord))

    def add(self, path_or_data: bytes | str, mimetype: str, coordinates: str, file_name: str | None=None, is_for_static_download: bool=False) -> str:
        """Add a new MediaFile with the given parameters and return its URL.

        If an identical file already exists, return the existing URL
        and registers the current session as a user.

        Safe to call from any thread.

        Parameters
        ----------
        path_or_data : bytes or str
            If bytes: the media file's raw data. If str: the name of a file
            to load from disk.
        mimetype : str
            The mime type for the file. E.g. "audio/mpeg".
            This string will be used in the "Content-Type" header when the file
            is served over HTTP.
        coordinates : str
            Unique string identifying an element's location.
            Prevents memory leak of "forgotten" file IDs when element media
            is being replaced-in-place (e.g. an st.image stream).
            coordinates should be of the form: "1.(3.-14).5"
        file_name : str or None
            Optional file_name. Used to set the filename in the response header.
        is_for_static_download: bool
            Indicate that data stored for downloading as a file,
            not as a media for rendering at page. [default: False]

        Returns
        -------
        str
            The url that the frontend can use to fetch the media.

        Raises
        ------
        If a filename is passed, any Exception raised when trying to read the
        file will be re-raised.
        """
        session_id = _get_session_id()
        with self._lock:
            kind = MediaFileKind.DOWNLOADABLE if is_for_static_download else MediaFileKind.MEDIA
            file_id = self._storage.load_and_get_id(path_or_data, mimetype, kind, file_name)
            metadata = MediaFileMetadata(kind=kind)
            self._file_metadata[file_id] = metadata
            self._files_by_session_and_coord[session_id][coordinates] = file_id
            return self._storage.get_url(file_id)