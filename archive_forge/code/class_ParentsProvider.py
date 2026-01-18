import os
import stat
import sys
import time
import warnings
from io import BytesIO
from typing import (
from .errors import (
from .file import GitFile
from .hooks import (
from .line_ending import BlobNormalizer, TreeBlobNormalizer
from .object_store import (
from .objects import (
from .pack import generate_unpacked_objects
from .refs import (
class ParentsProvider:

    def __init__(self, store, grafts={}, shallows=[]) -> None:
        self.store = store
        self.grafts = grafts
        self.shallows = set(shallows)

    def get_parents(self, commit_id, commit=None):
        try:
            return self.grafts[commit_id]
        except KeyError:
            pass
        if commit_id in self.shallows:
            return []
        if commit is None:
            commit = self.store[commit_id]
        return commit.parents