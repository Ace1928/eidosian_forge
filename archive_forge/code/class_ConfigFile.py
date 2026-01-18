import os
import sys
from contextlib import suppress
from typing import (
from .file import GitFile
class ConfigFile(ConfigDict):
    """A Git configuration file, like .git/config or ~/.gitconfig."""

    def __init__(self, values: Union[MutableMapping[Section, MutableMapping[Name, Value]], None]=None, encoding: Union[str, None]=None) -> None:
        super().__init__(values=values, encoding=encoding)
        self.path: Optional[str] = None

    @classmethod
    def from_file(cls, f: BinaryIO) -> 'ConfigFile':
        """Read configuration from a file-like object."""
        ret = cls()
        section: Optional[Section] = None
        setting = None
        continuation = None
        for lineno, line in enumerate(f.readlines()):
            if lineno == 0 and line.startswith(b'\xef\xbb\xbf'):
                line = line[3:]
            line = line.lstrip()
            if setting is None:
                if len(line) > 0 and line[:1] == b'[':
                    section, line = _parse_section_header_line(line)
                    ret._values.setdefault(section)
                if _strip_comments(line).strip() == b'':
                    continue
                if section is None:
                    raise ValueError('setting %r without section' % line)
                try:
                    setting, value = line.split(b'=', 1)
                except ValueError:
                    setting = line
                    value = b'true'
                setting = setting.strip()
                if not _check_variable_name(setting):
                    raise ValueError('invalid variable name %r' % setting)
                if value.endswith(b'\\\n'):
                    continuation = value[:-2]
                elif value.endswith(b'\\\r\n'):
                    continuation = value[:-3]
                else:
                    continuation = None
                    value = _parse_string(value)
                    ret._values[section][setting] = value
                    setting = None
            elif line.endswith(b'\\\n'):
                continuation += line[:-2]
            elif line.endswith(b'\\\r\n'):
                continuation += line[:-3]
            else:
                continuation += line
                value = _parse_string(continuation)
                ret._values[section][setting] = value
                continuation = None
                setting = None
        return ret

    @classmethod
    def from_path(cls, path: str) -> 'ConfigFile':
        """Read configuration from a file on disk."""
        with GitFile(path, 'rb') as f:
            ret = cls.from_file(f)
            ret.path = path
            return ret

    def write_to_path(self, path: Optional[str]=None) -> None:
        """Write configuration to a file on disk."""
        if path is None:
            path = self.path
        with GitFile(path, 'wb') as f:
            self.write_to_file(f)

    def write_to_file(self, f: BinaryIO) -> None:
        """Write configuration to a file-like object."""
        for section, values in self._values.items():
            try:
                section_name, subsection_name = section
            except ValueError:
                section_name, = section
                subsection_name = None
            if subsection_name is None:
                f.write(b'[' + section_name + b']\n')
            else:
                f.write(b'[' + section_name + b' "' + subsection_name + b'"]\n')
            for key, value in values.items():
                value = _format_string(value)
                f.write(b'\t' + key + b' = ' + value + b'\n')