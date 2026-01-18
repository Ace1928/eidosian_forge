import os
import sys
import warnings
from typing import ClassVar, Set
class FileLocked(Exception):
    """File is already locked."""

    def __init__(self, filename, lockfilename) -> None:
        self.filename = filename
        self.lockfilename = lockfilename
        super().__init__(filename, lockfilename)