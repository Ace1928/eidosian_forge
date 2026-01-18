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
def _topo_iter(parents, versions):
    seen = set()
    descendants = {}

    def pending_parents(version):
        if parents[version] is None:
            return []
        return [v for v in parents[version] if v in versions and v not in seen]
    for version_id in versions:
        if parents[version_id] is None:
            continue
        for parent_id in parents[version_id]:
            descendants.setdefault(parent_id, []).append(version_id)
    cur = [v for v in versions if len(pending_parents(v)) == 0]
    while len(cur) > 0:
        next = []
        for version_id in cur:
            if version_id in seen:
                continue
            if len(pending_parents(version_id)) != 0:
                continue
            next.extend(descendants.get(version_id, []))
            yield version_id
            seen.add(version_id)
        cur = next