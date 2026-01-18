import contextlib
import itertools
import re
import sys
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set, Tuple
from . import branch as _mod_branch
from . import errors
from .inter import InterObject
from .registry import Registry
from .revision import RevisionID
class MemoryTags(Tags):

    def __init__(self, tag_dict):
        self._tag_dict = tag_dict

    def get_tag_dict(self):
        return self._tag_dict

    def lookup_tag(self, tag_name):
        """Return the referent string of a tag"""
        td = self.get_tag_dict()
        try:
            return td[tag_name]
        except KeyError:
            raise errors.NoSuchTag(tag_name)

    def set_tag(self, name, revid):
        self._tag_dict[name] = revid

    def delete_tag(self, name):
        try:
            del self._tag_dict[name]
        except KeyError:
            raise errors.NoSuchTag(name)

    def rename_revisions(self, revid_map):
        self._tag_dict = {name: revid_map.get(revid, revid) for name, revid in self._tag_dict.items()}

    def _set_tag_dict(self, result):
        self._tag_dict = dict(result.items())

    def merge_to(self, to_tags, overwrite=False, ignore_master=False, selector=None):
        source_dict = self.get_tag_dict()
        dest_dict = to_tags.get_tag_dict()
        result, updates, conflicts = _reconcile_tags(source_dict, dest_dict, overwrite, selector)
        if result != dest_dict:
            to_tags._set_tag_dict(result)
        return (updates, conflicts)