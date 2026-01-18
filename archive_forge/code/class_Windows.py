from __future__ import annotations
import ctypes
import os
import sys
from functools import lru_cache
from typing import TYPE_CHECKING
from .api import PlatformDirsABC
class Windows(PlatformDirsABC):
    """
    `MSDN on where to store app data files
    <http://support.microsoft.com/default.aspx?scid=kb;en-us;310294#XSLTH3194121123120121120120>`_.
    Makes use of the
    `appname <platformdirs.api.PlatformDirsABC.appname>`,
    `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`,
    `version <platformdirs.api.PlatformDirsABC.version>`,
    `roaming <platformdirs.api.PlatformDirsABC.roaming>`,
    `opinion <platformdirs.api.PlatformDirsABC.opinion>`,
    `ensure_exists <platformdirs.api.PlatformDirsABC.ensure_exists>`.
    """

    @property
    def user_data_dir(self) -> str:
        """
        :return: data directory tied to the user, e.g.
         ``%USERPROFILE%\\AppData\\Local\\$appauthor\\$appname`` (not roaming) or
         ``%USERPROFILE%\\AppData\\Roaming\\$appauthor\\$appname`` (roaming)
        """
        const = 'CSIDL_APPDATA' if self.roaming else 'CSIDL_LOCAL_APPDATA'
        path = os.path.normpath(get_win_folder(const))
        return self._append_parts(path)

    def _append_parts(self, path: str, *, opinion_value: str | None=None) -> str:
        params = []
        if self.appname:
            if self.appauthor is not False:
                author = self.appauthor or self.appname
                params.append(author)
            params.append(self.appname)
            if opinion_value is not None and self.opinion:
                params.append(opinion_value)
            if self.version:
                params.append(self.version)
        path = os.path.join(path, *params)
        self._optionally_create_directory(path)
        return path

    @property
    def site_data_dir(self) -> str:
        """:return: data directory shared by users, e.g. ``C:\\ProgramData\\$appauthor\\$appname``"""
        path = os.path.normpath(get_win_folder('CSIDL_COMMON_APPDATA'))
        return self._append_parts(path)

    @property
    def user_config_dir(self) -> str:
        """:return: config directory tied to the user, same as `user_data_dir`"""
        return self.user_data_dir

    @property
    def site_config_dir(self) -> str:
        """:return: config directory shared by the users, same as `site_data_dir`"""
        return self.site_data_dir

    @property
    def user_cache_dir(self) -> str:
        """
        :return: cache directory tied to the user (if opinionated with ``Cache`` folder within ``$appname``) e.g.
         ``%USERPROFILE%\\AppData\\Local\\$appauthor\\$appname\\Cache\\$version``
        """
        path = os.path.normpath(get_win_folder('CSIDL_LOCAL_APPDATA'))
        return self._append_parts(path, opinion_value='Cache')

    @property
    def site_cache_dir(self) -> str:
        """:return: cache directory shared by users, e.g. ``C:\\ProgramData\\$appauthor\\$appname\\Cache\\$version``"""
        path = os.path.normpath(get_win_folder('CSIDL_COMMON_APPDATA'))
        return self._append_parts(path, opinion_value='Cache')

    @property
    def user_state_dir(self) -> str:
        """:return: state directory tied to the user, same as `user_data_dir`"""
        return self.user_data_dir

    @property
    def user_log_dir(self) -> str:
        """:return: log directory tied to the user, same as `user_data_dir` if not opinionated else ``Logs`` in it"""
        path = self.user_data_dir
        if self.opinion:
            path = os.path.join(path, 'Logs')
            self._optionally_create_directory(path)
        return path

    @property
    def user_documents_dir(self) -> str:
        """:return: documents directory tied to the user e.g. ``%USERPROFILE%\\Documents``"""
        return os.path.normpath(get_win_folder('CSIDL_PERSONAL'))

    @property
    def user_downloads_dir(self) -> str:
        """:return: downloads directory tied to the user e.g. ``%USERPROFILE%\\Downloads``"""
        return os.path.normpath(get_win_folder('CSIDL_DOWNLOADS'))

    @property
    def user_pictures_dir(self) -> str:
        """:return: pictures directory tied to the user e.g. ``%USERPROFILE%\\Pictures``"""
        return os.path.normpath(get_win_folder('CSIDL_MYPICTURES'))

    @property
    def user_videos_dir(self) -> str:
        """:return: videos directory tied to the user e.g. ``%USERPROFILE%\\Videos``"""
        return os.path.normpath(get_win_folder('CSIDL_MYVIDEO'))

    @property
    def user_music_dir(self) -> str:
        """:return: music directory tied to the user e.g. ``%USERPROFILE%\\Music``"""
        return os.path.normpath(get_win_folder('CSIDL_MYMUSIC'))

    @property
    def user_runtime_dir(self) -> str:
        """
        :return: runtime directory tied to the user, e.g.
         ``%USERPROFILE%\\AppData\\Local\\Temp\\$appauthor\\$appname``
        """
        path = os.path.normpath(os.path.join(get_win_folder('CSIDL_LOCAL_APPDATA'), 'Temp'))
        return self._append_parts(path)