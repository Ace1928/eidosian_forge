from __future__ import annotations
import collections
import os
import sys
import types
from pathlib import Path
from typing import Callable, Final
from streamlit import config, file_util
from streamlit.folder_black_list import FolderBlackList
from streamlit.logger import get_logger
from streamlit.source_util import get_pages
from streamlit.watcher.path_watcher import (
class LocalSourcesWatcher:

    def __init__(self, main_script_path: str):
        self._main_script_path = os.path.abspath(main_script_path)
        self._script_folder = os.path.dirname(self._main_script_path)
        self._on_file_changed: list[Callable[[str], None]] = []
        self._is_closed = False
        self._cached_sys_modules: set[str] = set()
        self._folder_black_list = FolderBlackList(config.get_option('server.folderWatchBlacklist'))
        self._watched_modules: dict[str, WatchedModule] = {}
        self._watched_pages: set[str] = set()
        self.update_watched_pages()

    def update_watched_pages(self) -> None:
        old_watched_pages = self._watched_pages
        new_pages_paths: set[str] = set()
        for page_info in get_pages(self._main_script_path).values():
            new_pages_paths.add(page_info['script_path'])
            if page_info['script_path'] not in old_watched_pages:
                self._register_watcher(page_info['script_path'], module_name=None)
        for old_page_path in old_watched_pages:
            if old_page_path not in new_pages_paths:
                self._deregister_watcher(old_page_path)
        self._watched_pages = new_pages_paths

    def register_file_change_callback(self, cb: Callable[[str], None]) -> None:
        self._on_file_changed.append(cb)

    def on_file_changed(self, filepath):
        if filepath not in self._watched_modules:
            _LOGGER.error('Received event for non-watched file: %s', filepath)
            return
        for wm in self._watched_modules.values():
            if wm.module_name is not None and wm.module_name in sys.modules:
                del sys.modules[wm.module_name]
        for cb in self._on_file_changed:
            cb(filepath)

    def close(self):
        for wm in self._watched_modules.values():
            wm.watcher.close()
        self._watched_modules = {}
        self._watched_pages = set()
        self._is_closed = True

    def _register_watcher(self, filepath, module_name):
        global PathWatcher
        if PathWatcher is None:
            PathWatcher = get_default_path_watcher_class()
        if PathWatcher is NoOpPathWatcher:
            return
        try:
            wm = WatchedModule(watcher=PathWatcher(filepath, self.on_file_changed), module_name=module_name)
        except PermissionError:
            return
        self._watched_modules[filepath] = wm

    def _deregister_watcher(self, filepath):
        if filepath not in self._watched_modules:
            return
        if filepath == self._main_script_path:
            return
        wm = self._watched_modules[filepath]
        wm.watcher.close()
        del self._watched_modules[filepath]

    def _file_is_new(self, filepath):
        return filepath not in self._watched_modules

    def _file_should_be_watched(self, filepath):
        return self._file_is_new(filepath) and (file_util.file_is_in_folder_glob(filepath, self._script_folder) or file_util.file_in_pythonpath(filepath))

    def update_watched_modules(self):
        if self._is_closed:
            return
        if set(sys.modules) != self._cached_sys_modules:
            modules_paths = {name: self._exclude_blacklisted_paths(get_module_paths(module)) for name, module in dict(sys.modules).items()}
            self._cached_sys_modules = set(sys.modules)
            self._register_necessary_watchers(modules_paths)

    def _register_necessary_watchers(self, module_paths: dict[str, set[str]]) -> None:
        for name, paths in module_paths.items():
            for path in paths:
                if self._file_should_be_watched(path):
                    self._register_watcher(str(Path(path).resolve()), name)

    def _exclude_blacklisted_paths(self, paths: set[str]) -> set[str]:
        return {p for p in paths if not self._folder_black_list.is_blacklisted(p)}