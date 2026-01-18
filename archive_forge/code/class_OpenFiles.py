from __future__ import annotations
import io
import logging
import os
import re
from glob import has_magic
from pathlib import Path
from .caching import (  # noqa: F401
from .compression import compr
from .registry import filesystem, get_filesystem_class
from .utils import (
class OpenFiles(list):
    """List of OpenFile instances

    Can be used in a single context, which opens and closes all of the
    contained files. Normal list access to get the elements works as
    normal.

    A special case is made for caching filesystems - the files will
    be down/uploaded together at the start or end of the context, and
    this may happen concurrently, if the target filesystem supports it.
    """

    def __init__(self, *args, mode='rb', fs=None):
        self.mode = mode
        self.fs = fs
        self.files = []
        super().__init__(*args)

    def __enter__(self):
        if self.fs is None:
            raise ValueError('Context has already been used')
        fs = self.fs
        while True:
            if hasattr(fs, 'open_many'):
                self.files = fs.open_many(self)
                return self.files
            if hasattr(fs, 'fs') and fs.fs is not None:
                fs = fs.fs
            else:
                break
        return [s.__enter__() for s in self]

    def __exit__(self, *args):
        fs = self.fs
        [s.__exit__(*args) for s in self]
        if 'r' not in self.mode:
            while True:
                if hasattr(fs, 'open_many'):
                    fs.commit_many(self.files)
                    return
                if hasattr(fs, 'fs') and fs.fs is not None:
                    fs = fs.fs
                else:
                    break

    def __getitem__(self, item):
        out = super().__getitem__(item)
        if isinstance(item, slice):
            return OpenFiles(out, mode=self.mode, fs=self.fs)
        return out

    def __repr__(self):
        return f'<List of {len(self)} OpenFile instances>'