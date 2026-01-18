import os.path
import re
from contextlib import suppress
from typing import TYPE_CHECKING, BinaryIO, Dict, Iterable, List, Optional, Union
from .config import Config, get_xdg_config_home_path
class IgnoreFilterManager:
    """Ignore file manager."""

    def __init__(self, top_path: str, global_filters: List[IgnoreFilter], ignorecase: bool) -> None:
        self._path_filters: Dict[str, Optional[IgnoreFilter]] = {}
        self._top_path = top_path
        self._global_filters = global_filters
        self._ignorecase = ignorecase

    def __repr__(self) -> str:
        return '{}({}, {!r}, {!r})'.format(type(self).__name__, self._top_path, self._global_filters, self._ignorecase)

    def _load_path(self, path: str) -> Optional[IgnoreFilter]:
        try:
            return self._path_filters[path]
        except KeyError:
            pass
        p = os.path.join(self._top_path, path, '.gitignore')
        try:
            self._path_filters[path] = IgnoreFilter.from_path(p, self._ignorecase)
        except OSError:
            self._path_filters[path] = None
        return self._path_filters[path]

    def find_matching(self, path: str) -> Iterable[Pattern]:
        """Find matching patterns for path.

        Args:
          path: Path to check
        Returns:
          Iterator over Pattern instances
        """
        if os.path.isabs(path):
            raise ValueError('%s is an absolute path' % path)
        filters = [(0, f) for f in self._global_filters]
        if os.path.sep != '/':
            path = path.replace(os.path.sep, '/')
        parts = path.split('/')
        matches = []
        for i in range(len(parts) + 1):
            dirname = '/'.join(parts[:i])
            for s, f in filters:
                relpath = '/'.join(parts[s:i])
                if i < len(parts):
                    relpath += '/'
                matches += list(f.find_matching(relpath))
            ignore_filter = self._load_path(dirname)
            if ignore_filter is not None:
                filters.insert(0, (i, ignore_filter))
        return iter(matches)

    def is_ignored(self, path: str) -> Optional[bool]:
        """Check whether a path is explicitly included or excluded in ignores.

        Args:
          path: Path to check
        Returns:
          None if the file is not mentioned, True if it is included,
          False if it is explicitly excluded.
        """
        matches = list(self.find_matching(path))
        if matches:
            return matches[-1].is_exclude
        return None

    @classmethod
    def from_repo(cls, repo: 'Repo') -> 'IgnoreFilterManager':
        """Create a IgnoreFilterManager from a repository.

        Args:
          repo: Repository object
        Returns:
          A `IgnoreFilterManager` object
        """
        global_filters = []
        for p in [os.path.join(repo.controldir(), 'info', 'exclude'), default_user_ignore_filter_path(repo.get_config_stack())]:
            with suppress(OSError):
                global_filters.append(IgnoreFilter.from_path(os.path.expanduser(p)))
        config = repo.get_config_stack()
        ignorecase = config.get_boolean(b'core', b'ignorecase', False)
        return cls(repo.path, global_filters, ignorecase)