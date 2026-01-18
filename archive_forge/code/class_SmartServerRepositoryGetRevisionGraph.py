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
class SmartServerRepositoryGetRevisionGraph(SmartServerRepositoryReadLocked):

    def do_readlocked_repository_request(self, repository, revision_id):
        """Return the result of repository.get_revision_graph(revision_id).

        Deprecated as of bzr 1.4, but supported for older clients.

        :param repository: The repository to query in.
        :param revision_id: The utf8 encoded revision_id to get a graph from.
        :return: A smart server response where the body contains an utf8
            encoded flattened list of the revision graph.
        """
        if not revision_id:
            revision_id = None
        lines = []
        graph = repository.get_graph()
        if revision_id:
            search_ids = [revision_id]
        else:
            search_ids = repository.all_revision_ids()
        search = graph._make_breadth_first_searcher(search_ids)
        transitive_ids = set(itertools.chain.from_iterable(search))
        parent_map = graph.get_parent_map(transitive_ids)
        revision_graph = _strip_NULL_ghosts(parent_map)
        if revision_id and revision_id not in revision_graph:
            return FailedSmartServerResponse((b'nosuchrevision', revision_id), b'')
        for revision, parents in revision_graph.items():
            lines.append(b' '.join((revision,) + tuple(parents)))
        return SuccessfulSmartServerResponse((b'ok',), b'\n'.join(lines))