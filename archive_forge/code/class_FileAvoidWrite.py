import contextlib
import filecmp
import os
import re
import shutil
import sys
import unicodedata
from io import StringIO
from os import path
from typing import Any, Generator, Iterator, List, Optional, Type
class FileAvoidWrite:
    """File-like object that buffers output and only writes if content changed.

    Use this class like when writing to a file to avoid touching the original
    file if the content hasn't changed. This is useful in scenarios where file
    mtime is used to invalidate caches or trigger new behavior.

    When writing to this file handle, all writes are buffered until the object
    is closed.

    Objects can be used as context managers.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._io: Optional[StringIO] = None

    def write(self, data: str) -> None:
        if not self._io:
            self._io = StringIO()
        self._io.write(data)

    def close(self) -> None:
        """Stop accepting writes and write file, if needed."""
        if not self._io:
            raise Exception('FileAvoidWrite does not support empty files.')
        buf = self.getvalue()
        self._io.close()
        try:
            with open(self._path, encoding='utf-8') as old_f:
                old_content = old_f.read()
                if old_content == buf:
                    return
        except OSError:
            pass
        with open(self._path, 'w', encoding='utf-8') as f:
            f.write(buf)

    def __enter__(self) -> 'FileAvoidWrite':
        return self

    def __exit__(self, exc_type: Type[Exception], exc_value: Exception, traceback: Any) -> bool:
        self.close()
        return True

    def __getattr__(self, name: str) -> Any:
        if not self._io:
            raise Exception('Must write to FileAvoidWrite before other methods can be used')
        return getattr(self._io, name)