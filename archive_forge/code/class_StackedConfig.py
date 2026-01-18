import os
import sys
from contextlib import suppress
from typing import (
from .file import GitFile
class StackedConfig(Config):
    """Configuration which reads from multiple config files.."""

    def __init__(self, backends: List[ConfigFile], writable: Optional[ConfigFile]=None) -> None:
        self.backends = backends
        self.writable = writable

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} for {self.backends!r}>'

    @classmethod
    def default(cls) -> 'StackedConfig':
        return cls(cls.default_backends())

    @classmethod
    def default_backends(cls) -> List[ConfigFile]:
        """Retrieve the default configuration.

        See git-config(1) for details on the files searched.
        """
        paths = []
        paths.append(os.path.expanduser('~/.gitconfig'))
        paths.append(get_xdg_config_home_path('git', 'config'))
        if 'GIT_CONFIG_NOSYSTEM' not in os.environ:
            paths.append('/etc/gitconfig')
            if sys.platform == 'win32':
                paths.extend(get_win_system_paths())
        backends = []
        for path in paths:
            try:
                cf = ConfigFile.from_path(path)
            except FileNotFoundError:
                continue
            backends.append(cf)
        return backends

    def get(self, section: SectionLike, name: NameLike) -> Value:
        if not isinstance(section, tuple):
            section = (section,)
        for backend in self.backends:
            try:
                return backend.get(section, name)
            except KeyError:
                pass
        raise KeyError(name)

    def get_multivar(self, section: SectionLike, name: NameLike) -> Iterator[Value]:
        if not isinstance(section, tuple):
            section = (section,)
        for backend in self.backends:
            try:
                yield from backend.get_multivar(section, name)
            except KeyError:
                pass

    def set(self, section: SectionLike, name: NameLike, value: Union[ValueLike, bool]) -> None:
        if self.writable is None:
            raise NotImplementedError(self.set)
        return self.writable.set(section, name, value)

    def sections(self) -> Iterator[Section]:
        seen = set()
        for backend in self.backends:
            for section in backend.sections():
                if section not in seen:
                    seen.add(section)
                    yield section