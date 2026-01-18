import errno
import os
from io import BytesIO
from .lazy_import import lazy_import
import gzip
import itertools
import patiencediff
from breezy import (
from . import errors
from .i18n import gettext
class _Reconstructor:
    """Build a text from the diffs, ancestry graph and cached lines"""

    def __init__(self, diffs, lines, parents):
        self.diffs = diffs
        self.lines = lines
        self.parents = parents
        self.cursor = {}

    def reconstruct(self, lines, parent_text, version_id):
        """Append the lines referred to by a ParentText to lines"""
        parent_id = self.parents[version_id][parent_text.parent]
        end = parent_text.parent_pos + parent_text.num_lines
        return self._reconstruct(lines, parent_id, parent_text.parent_pos, end)

    def _reconstruct(self, lines, req_version_id, req_start, req_end):
        """Append lines for the requested version_id range"""
        if req_start == req_end:
            return
        pending_reqs = [(req_version_id, req_start, req_end)]
        while len(pending_reqs) > 0:
            req_version_id, req_start, req_end = pending_reqs.pop()
            if req_version_id in self.lines:
                lines.extend(self.lines[req_version_id][req_start:req_end])
                continue
            try:
                start, end, kind, data, iterator = self.cursor[req_version_id]
            except KeyError:
                iterator = self.diffs.get_diff(req_version_id).range_iterator()
                start, end, kind, data = next(iterator)
            if start > req_start:
                iterator = self.diffs.get_diff(req_version_id).range_iterator()
                start, end, kind, data = next(iterator)
            while end <= req_start:
                start, end, kind, data = next(iterator)
            self.cursor[req_version_id] = (start, end, kind, data, iterator)
            if req_end > end:
                pending_reqs.append((req_version_id, end, req_end))
                req_end = end
            if kind == 'new':
                lines.extend(data[req_start - start:req_end - start])
            else:
                parent, parent_start, parent_end = data
                new_version_id = self.parents[req_version_id][parent]
                new_start = parent_start + req_start - start
                new_end = parent_end + req_end - end
                pending_reqs.append((new_version_id, new_start, new_end))

    def reconstruct_version(self, lines, version_id):
        length = self.diffs.get_diff(version_id).num_lines()
        return self._reconstruct(lines, version_id, 0, length)