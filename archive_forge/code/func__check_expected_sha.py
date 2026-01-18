import posixpath
import stat
from typing import Dict, Iterable, Iterator, List
from dulwich.object_store import BaseObjectStore
from dulwich.objects import (ZERO_SHA, Blob, Commit, ObjectID, ShaFile, Tree,
from dulwich.pack import Pack, PackData, pack_objects_to_data
from .. import errors, lru_cache, osutils, trace, ui
from ..bzr.testament import StrictTestament3
from ..lock import LogicalLockResult
from ..revision import NULL_REVISION
from ..tree import InterTree
from .cache import from_repository as cache_from_repository
from .mapping import (default_mapping, encode_git_path, entry_mode,
from .unpeel_map import UnpeelMap
def _check_expected_sha(expected_sha, object):
    """Check whether an object matches an expected SHA.

    :param expected_sha: None or expected SHA as either binary or as hex digest
    :param object: Object to verify
    """
    if expected_sha is None:
        return
    if len(expected_sha) == 40:
        if expected_sha != object.sha().hexdigest().encode('ascii'):
            raise AssertionError('Invalid sha for {!r}: {}'.format(object, expected_sha))
    elif len(expected_sha) == 20:
        if expected_sha != object.sha().digest():
            raise AssertionError('Invalid sha for {!r}: {}'.format(object, sha_to_hex(expected_sha)))
    else:
        raise AssertionError('Unknown length %d for %r' % (len(expected_sha), expected_sha))