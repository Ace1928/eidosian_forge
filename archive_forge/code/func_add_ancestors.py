from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def add_ancestors(file_id):
    if not byid.has_id(file_id):
        return
    parent_id = byid.get_entry(file_id).parent_id
    if parent_id is None:
        return
    if parent_id not in parents:
        parents.add(parent_id)
        add_ancestors(parent_id)