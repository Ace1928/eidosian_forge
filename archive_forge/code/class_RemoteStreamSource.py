import bz2
import os
import re
import sys
import zlib
from typing import Callable, List, Optional
import fastbencode as bencode
from .. import branch
from .. import bzr as _mod_bzr
from .. import config as _mod_config
from .. import (controldir, debug, errors, gpg, graph, lock, lockdir, osutils,
from .. import repository as _mod_repository
from .. import revision as _mod_revision
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..branch import BranchWriteLockResult
from ..decorators import only_raises
from ..errors import NoSuchRevision, SmartProtocolError
from ..i18n import gettext
from ..lockable_files import LockableFiles
from ..repository import RepositoryWriteLockResult, _LazyListJoin
from ..revision import NULL_REVISION
from ..trace import log_exception_quietly, mutter, note, warning
from . import branch as bzrbranch
from . import bzrdir as _mod_bzrdir
from . import inventory_delta
from . import repository as bzrrepository
from . import testament as _mod_testament
from . import vf_repository, vf_search
from .branch import BranchReferenceFormat
from .inventory import Inventory
from .inventorytree import InventoryRevisionTree
from .serializer import format_registry as serializer_format_registry
from .smart import client
from .smart import repository as smart_repo
from .smart import vfs
from .smart.client import _SmartClient
from .versionedfile import FulltextContentFactory
class RemoteStreamSource(vf_repository.StreamSource):
    """Stream data from a remote server."""

    def get_stream(self, search):
        if self.from_repository._fallback_repositories and self.to_format._fetch_order == 'topological':
            return self._real_stream(self.from_repository, search)
        sources = []
        seen = set()
        repos = [self.from_repository]
        while repos:
            repo = repos.pop(0)
            if repo in seen:
                continue
            seen.add(repo)
            repos.extend(repo._fallback_repositories)
            sources.append(repo)
        return self.missing_parents_chain(search, sources)

    def _get_real_stream_for_missing_keys(self, missing_keys):
        self.from_repository._ensure_real()
        real_repo = self.from_repository._real_repository
        real_source = real_repo._get_source(self.to_format)
        return real_source.get_stream_for_missing_keys(missing_keys)

    def get_stream_for_missing_keys(self, missing_keys):
        if not isinstance(self.from_repository, RemoteRepository):
            return self._get_real_stream_for_missing_keys(missing_keys)
        client = self.from_repository._client
        medium = client._medium
        if medium._is_remote_before((3, 0)):
            return self._get_real_stream_for_missing_keys(missing_keys)
        path = self.from_repository.controldir._path_for_remote_call(client)
        args = (path, self.to_format.network_name())
        search_bytes = b'\n'.join([b'%s\t%s' % (key[0].encode('utf-8'), key[1]) for key in missing_keys])
        try:
            response, handler = self.from_repository._call_with_body_bytes_expecting_body(b'Repository.get_stream_for_missing_keys', args, search_bytes)
        except (errors.UnknownSmartMethod, errors.UnknownFormatError):
            return self._get_real_stream_for_missing_keys(missing_keys)
        if response[0] != b'ok':
            raise errors.UnexpectedSmartServerResponse(response)
        byte_stream = handler.read_streamed_body()
        src_format, stream = smart_repo._byte_stream_to_stream(byte_stream, self._record_counter)
        if src_format.network_name() != self.from_repository._format.network_name():
            raise AssertionError('Mismatched RemoteRepository and stream src {!r}, {!r}'.format(src_format.network_name(), repo._format.network_name()))
        return stream

    def _real_stream(self, repo, search):
        """Get a stream for search from repo.

        This never called RemoteStreamSource.get_stream, and is a helper
        for RemoteStreamSource._get_stream to allow getting a stream
        reliably whether fallback back because of old servers or trying
        to stream from a non-RemoteRepository (which the stacked support
        code will do).
        """
        source = repo._get_source(self.to_format)
        if isinstance(source, RemoteStreamSource):
            repo._ensure_real()
            source = repo._real_repository._get_source(self.to_format)
        return source.get_stream(search)

    def _get_stream(self, repo, search):
        """Core worker to get a stream from repo for search.

        This is used by both get_stream and the stacking support logic. It
        deliberately gets a stream for repo which does not need to be
        self.from_repository. In the event that repo is not Remote, or
        cannot do a smart stream, a fallback is made to the generic
        repository._get_stream() interface, via self._real_stream.

        In the event of stacking, streams from _get_stream will not
        contain all the data for search - this is normal (see get_stream).

        :param repo: A repository.
        :param search: A search.
        """
        if not isinstance(repo, RemoteRepository):
            return self._real_stream(repo, search)
        client = repo._client
        medium = client._medium
        path = repo.controldir._path_for_remote_call(client)
        search_bytes = repo._serialise_search_result(search)
        args = (path, self.to_format.network_name())
        candidate_verbs = [(b'Repository.get_stream_1.19', (1, 19)), (b'Repository.get_stream', (1, 13))]
        found_verb = False
        for verb, version in candidate_verbs:
            if medium._is_remote_before(version):
                continue
            try:
                response = repo._call_with_body_bytes_expecting_body(verb, args, search_bytes)
            except errors.UnknownSmartMethod:
                medium._remember_remote_is_before(version)
            except UnknownErrorFromSmartServer as e:
                if isinstance(search, vf_search.EverythingResult):
                    error_verb = e.error_from_smart_server.error_verb
                    if error_verb == b'BadSearch':
                        medium._remember_remote_is_before((2, 4))
                        break
                raise
            else:
                response_tuple, response_handler = response
                found_verb = True
                break
        if not found_verb:
            return self._real_stream(repo, search)
        if response_tuple[0] != b'ok':
            raise errors.UnexpectedSmartServerResponse(response_tuple)
        byte_stream = response_handler.read_streamed_body()
        src_format, stream = smart_repo._byte_stream_to_stream(byte_stream, self._record_counter)
        if src_format.network_name() != repo._format.network_name():
            raise AssertionError('Mismatched RemoteRepository and stream src {!r}, {!r}'.format(src_format.network_name(), repo._format.network_name()))
        return stream

    def missing_parents_chain(self, search, sources):
        """Chain multiple streams together to handle stacking.

        :param search: The overall search to satisfy with streams.
        :param sources: A list of Repository objects to query.
        """
        self.from_serialiser = self.from_repository._format._serializer
        self.seen_revs = set()
        self.referenced_revs = set()
        while not search.is_empty() and len(sources) > 1:
            source = sources.pop(0)
            stream = self._get_stream(source, search)
            for kind, substream in stream:
                if kind != 'revisions':
                    yield (kind, substream)
                else:
                    yield (kind, self.missing_parents_rev_handler(substream))
            search = search.refine(self.seen_revs, self.referenced_revs)
            self.seen_revs = set()
            self.referenced_revs = set()
        if not search.is_empty():
            for kind, stream in self._get_stream(sources[0], search):
                yield (kind, stream)

    def missing_parents_rev_handler(self, substream):
        for content in substream:
            revision_bytes = content.get_bytes_as('fulltext')
            revision = self.from_serialiser.read_revision_from_string(revision_bytes)
            self.seen_revs.add(content.key[-1])
            self.referenced_revs.update(revision.parent_ids)
            yield content