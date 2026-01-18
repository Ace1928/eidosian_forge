import bz2
import itertools
import os
import queue
import sys
import tempfile
import threading
import zlib
import fastbencode as bencode
from ... import errors, estimate_compressed_size, osutils
from ... import revision as _mod_revision
from ... import trace, ui
from ...repository import _strip_NULL_ghosts, network_format_registry
from .. import inventory as _mod_inventory
from .. import inventory_delta, pack, vf_search
from ..bzrdir import BzrDir
from ..versionedfile import (ChunkedContentFactory, NetworkRecordStream,
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerRepositoryGetParentMap(SmartServerRepositoryRequest):
    """Bzr 1.2+ - get parent data for revisions during a graph search."""
    no_extra_results = False

    def do_repository_request(self, repository, *revision_ids):
        """Get parent details for some revisions.

        All the parents for revision_ids are returned. Additionally up to 64KB
        of additional parent data found by performing a breadth first search
        from revision_ids is returned. The verb takes a body containing the
        current search state, see do_body for details.

        If 'include-missing:' is in revision_ids, ghosts encountered in the
        graph traversal for getting parent data are included in the result with
        a prefix of 'missing:'.

        :param repository: The repository to query in.
        :param revision_ids: The utf8 encoded revision_id to answer for.
        """
        self._revision_ids = revision_ids
        return None

    def do_body(self, body_bytes):
        """Process the current search state and perform the parent lookup.

        :return: A smart server response where the body contains an utf8
            encoded flattened list of the parents of the revisions (the same
            format as Repository.get_revision_graph) which has been bz2
            compressed.
        """
        repository = self._repository
        with repository.lock_read():
            return self._do_repository_request(body_bytes)

    def _expand_requested_revs(self, repo_graph, revision_ids, client_seen_revs, include_missing, max_size=65536):
        result = {}
        queried_revs = set()
        estimator = estimate_compressed_size.ZLibEstimator(max_size)
        next_revs = revision_ids
        first_loop_done = False
        while next_revs:
            queried_revs.update(next_revs)
            parent_map = repo_graph.get_parent_map(next_revs)
            current_revs = next_revs
            next_revs = set()
            for revision_id in current_revs:
                missing_rev = False
                parents = parent_map.get(revision_id)
                if parents is not None:
                    if parents == (_mod_revision.NULL_REVISION,):
                        parents = ()
                    next_revs.update(parents)
                    encoded_id = revision_id
                else:
                    missing_rev = True
                    encoded_id = b'missing:' + revision_id
                    parents = []
                if revision_id not in client_seen_revs and (not missing_rev or include_missing):
                    result[encoded_id] = parents
                    line = encoded_id + b' ' + b' '.join(parents) + b'\n'
                    estimator.add_content(line)
            if self.no_extra_results or (first_loop_done and estimator.full()):
                trace.mutter('size: %d, z_size: %d' % (estimator._uncompressed_size_added, estimator._compressed_size_added))
                next_revs = set()
                break
            next_revs = next_revs.difference(queried_revs)
            first_loop_done = True
        return result

    def _do_repository_request(self, body_bytes):
        repository = self._repository
        revision_ids = set(self._revision_ids)
        include_missing = b'include-missing:' in revision_ids
        if include_missing:
            revision_ids.remove(b'include-missing:')
        body_lines = body_bytes.split(b'\n')
        search_result, error = self.recreate_search_from_recipe(repository, body_lines)
        if error is not None:
            return error
        client_seen_revs = set(search_result.get_keys())
        client_seen_revs.difference_update(revision_ids)
        repo_graph = repository.get_graph()
        result = self._expand_requested_revs(repo_graph, revision_ids, client_seen_revs, include_missing)
        lines = []
        for revision, parents in sorted(result.items()):
            lines.append(b' '.join((revision,) + tuple(parents)))
        return SuccessfulSmartServerResponse((b'ok',), bz2.compress(b'\n'.join(lines)))