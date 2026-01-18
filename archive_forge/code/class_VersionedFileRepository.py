from io import BytesIO
from ..lazy_import import lazy_import
import itertools
from breezy import (
from breezy.bzr import (
from breezy.bzr.bundle import serializer
from breezy.i18n import gettext
from breezy.bzr.testament import Testament
from .. import errors
from ..decorators import only_raises
from ..repository import (CommitBuilder, FetchResult, InterRepository,
from ..trace import mutter, note
from .inventory import ROOT_ID, Inventory, entry_factory
from .inventorytree import InventoryTreeChange
from .repository import MetaDirRepository, RepositoryFormatMetaDir
class VersionedFileRepository(Repository):
    """Repository holding history for one or more branches.

    The repository holds and retrieves historical information including
    revisions and file history.  It's normally accessed only by the Branch,
    which views a particular line of development through that history.

    The Repository builds on top of some byte storage facilies (the revisions,
    signatures, inventories, texts and chk_bytes attributes) and a Transport,
    which respectively provide byte storage and a means to access the (possibly
    remote) disk.

    The byte storage facilities are addressed via tuples, which we refer to
    as 'keys' throughout the code base. Revision_keys, inventory_keys and
    signature_keys are all 1-tuples: (revision_id,). text_keys are two-tuples:
    (file_id, revision_id). chk_bytes uses CHK keys - a 1-tuple with a single
    byte string made up of a hash identifier and a hash value.
    We use this interface because it allows low friction with the underlying
    code that implements disk indices, network encoding and other parts of
    breezy.

    :ivar revisions: A breezy.versionedfile.VersionedFiles instance containing
        the serialised revisions for the repository. This can be used to obtain
        revision graph information or to access raw serialised revisions.
        The result of trying to insert data into the repository via this store
        is undefined: it should be considered read-only except for implementors
        of repositories.
    :ivar signatures: A breezy.versionedfile.VersionedFiles instance containing
        the serialised signatures for the repository. This can be used to
        obtain access to raw serialised signatures.  The result of trying to
        insert data into the repository via this store is undefined: it should
        be considered read-only except for implementors of repositories.
    :ivar inventories: A breezy.versionedfile.VersionedFiles instance containing
        the serialised inventories for the repository. This can be used to
        obtain unserialised inventories.  The result of trying to insert data
        into the repository via this store is undefined: it should be
        considered read-only except for implementors of repositories.
    :ivar texts: A breezy.versionedfile.VersionedFiles instance containing the
        texts of files and directories for the repository. This can be used to
        obtain file texts or file graphs. Note that Repository.iter_file_bytes
        is usually a better interface for accessing file texts.
        The result of trying to insert data into the repository via this store
        is undefined: it should be considered read-only except for implementors
        of repositories.
    :ivar chk_bytes: A breezy.versionedfile.VersionedFiles instance containing
        any data the repository chooses to store or have indexed by its hash.
        The result of trying to insert data into the repository via this store
        is undefined: it should be considered read-only except for implementors
        of repositories.
    :ivar _transport: Transport for file access to repository, typically
        pointing to .bzr/repository.
    """
    _commit_builder_class = VersionedFileCommitBuilder

    def add_fallback_repository(self, repository):
        """Add a repository to use for looking up data not held locally.

        :param repository: A repository.
        """
        if not self._format.supports_external_lookups:
            raise errors.UnstackableRepositoryFormat(self._format, self.base)
        self._check_fallback_repository(repository)
        if self.is_locked():
            repository.lock_read()
        self._fallback_repositories.append(repository)
        self.texts.add_fallback_versioned_files(repository.texts)
        self.inventories.add_fallback_versioned_files(repository.inventories)
        self.revisions.add_fallback_versioned_files(repository.revisions)
        self.signatures.add_fallback_versioned_files(repository.signatures)
        if self.chk_bytes is not None:
            self.chk_bytes.add_fallback_versioned_files(repository.chk_bytes)

    def create_bundle(self, target, base, fileobj, format=None):
        return serializer.write_bundle(self, target, base, fileobj, format)

    @only_raises(errors.LockNotHeld, errors.LockBroken)
    def unlock(self):
        super().unlock()
        if self.control_files._lock_count == 0:
            self._inventory_entry_cache.clear()

    def add_inventory(self, revision_id, inv, parents):
        """Add the inventory inv to the repository as revision_id.

        :param parents: The revision ids of the parents that revision_id
                        is known to have and are in the repository already.

        :returns: The validator(which is a sha1 digest, though what is sha'd is
            repository format specific) of the serialized inventory.
        """
        if not self.is_in_write_group():
            raise AssertionError('{!r} not in write group'.format(self))
        _mod_revision.check_not_reserved_id(revision_id)
        if not (inv.revision_id is None or inv.revision_id == revision_id):
            raise AssertionError('Mismatch between inventory revision id and insertion revid (%r, %r)' % (inv.revision_id, revision_id))
        if inv.root is None:
            raise errors.RootMissing()
        return self._add_inventory_checked(revision_id, inv, parents)

    def _add_inventory_checked(self, revision_id, inv, parents):
        """Add inv to the repository after checking the inputs.

        This function can be overridden to allow different inventory styles.

        :seealso: add_inventory, for the contract.
        """
        inv_lines = self._serializer.write_inventory_to_lines(inv)
        return self._inventory_add_lines(revision_id, parents, inv_lines, check_content=False)

    def add_inventory_by_delta(self, basis_revision_id, delta, new_revision_id, parents, basis_inv=None, propagate_caches=False):
        """Add a new inventory expressed as a delta against another revision.

        See the inventory developers documentation for the theory behind
        inventory deltas.

        :param basis_revision_id: The inventory id the delta was created
            against. (This does not have to be a direct parent.)
        :param delta: The inventory delta (see Inventory.apply_delta for
            details).
        :param new_revision_id: The revision id that the inventory is being
            added for.
        :param parents: The revision ids of the parents that revision_id is
            known to have and are in the repository already. These are supplied
            for repositories that depend on the inventory graph for revision
            graph access, as well as for those that pun ancestry with delta
            compression.
        :param basis_inv: The basis inventory if it is already known,
            otherwise None.
        :param propagate_caches: If True, the caches for this inventory are
          copied to and updated for the result if possible.

        :returns: (validator, new_inv)
            The validator(which is a sha1 digest, though what is sha'd is
            repository format specific) of the serialized inventory, and the
            resulting inventory.
        """
        if not self.is_in_write_group():
            raise AssertionError('{!r} not in write group'.format(self))
        _mod_revision.check_not_reserved_id(new_revision_id)
        basis_tree = self.revision_tree(basis_revision_id)
        with basis_tree.lock_read():
            if basis_inv is None:
                basis_inv = basis_tree.root_inventory
            basis_inv.apply_delta(delta)
            basis_inv.revision_id = new_revision_id
            return (self.add_inventory(new_revision_id, basis_inv, parents), basis_inv)

    def _inventory_add_lines(self, revision_id, parents, lines, check_content=True):
        """Store lines in inv_vf and return the sha1 of the inventory."""
        parents = [(parent,) for parent in parents]
        result = self.inventories.add_lines((revision_id,), parents, lines, check_content=check_content)[0]
        self.inventories._access.flush()
        return result

    def add_revision(self, revision_id, rev, inv=None):
        """Add rev to the revision store as revision_id.

        :param revision_id: the revision id to use.
        :param rev: The revision object.
        :param inv: The inventory for the revision. if None, it will be looked
                    up in the inventory storer
        """
        _mod_revision.check_not_reserved_id(revision_id)
        if not self.inventories.get_parent_map([(revision_id,)]):
            if inv is None:
                raise errors.WeaveRevisionNotPresent(revision_id, self.inventories)
            else:
                rev.inventory_sha1 = self.add_inventory(revision_id, inv, rev.parent_ids)
        else:
            key = (revision_id,)
            rev.inventory_sha1 = self.inventories.get_sha1s([key])[key]
        self._add_revision(rev)

    def _add_revision(self, revision):
        lines = self._serializer.write_revision_to_lines(revision)
        key = (revision.revision_id,)
        parents = tuple(((parent,) for parent in revision.parent_ids))
        self.revisions.add_lines(key, parents, lines)

    def _check_inventories(self, checker):
        """Check the inventories found from the revision scan.

        This is responsible for verifying the sha1 of inventories and
        creating a pending_keys set that covers data referenced by inventories.
        """
        with ui.ui_factory.nested_progress_bar() as bar:
            self._do_check_inventories(checker, bar)

    def _do_check_inventories(self, checker, bar):
        """Helper for _check_inventories."""
        revno = 0
        keys = {'chk_bytes': set(), 'inventories': set(), 'texts': set()}
        kinds = ['chk_bytes', 'texts']
        count = len(checker.pending_keys)
        bar.update(gettext('inventories'), 0, 2)
        current_keys = checker.pending_keys
        checker.pending_keys = {}
        for key in current_keys:
            if key[0] != 'inventories' and key[0] not in kinds:
                checker._report_items.append('unknown key type {!r}'.format(key))
            keys[key[0]].add(key[1:])
        if keys['inventories']:
            last_object = None
            for record in self.inventories.check(keys=keys['inventories']):
                if record.storage_kind == 'absent':
                    checker._report_items.append('Missing inventory {{{}}}'.format(record.key))
                else:
                    last_object = self._check_record('inventories', record, checker, last_object, current_keys[('inventories',) + record.key])
            del keys['inventories']
        else:
            return
        bar.update(gettext('texts'), 1)
        while checker.pending_keys or keys['chk_bytes'] or keys['texts']:
            current_keys = checker.pending_keys
            checker.pending_keys = {}
            for key in current_keys:
                if key[0] not in kinds:
                    checker._report_items.append('unknown key type {!r}'.format(key))
                keys[key[0]].add(key[1:])
            for kind in kinds:
                if keys[kind]:
                    last_object = None
                    for record in getattr(self, kind).check(keys=keys[kind]):
                        if record.storage_kind == 'absent':
                            checker._report_items.append('Missing {} {{{}}}'.format(kind, record.key))
                        else:
                            last_object = self._check_record(kind, record, checker, last_object, current_keys[(kind,) + record.key])
                    keys[kind] = set()
                    break

    def _check_record(self, kind, record, checker, last_object, item_data):
        """Check a single text from this repository."""
        if kind == 'inventories':
            rev_id = record.key[0]
            inv = self._deserialise_inventory(rev_id, record.get_bytes_as('lines'))
            if last_object is not None:
                delta = inv._make_delta(last_object)
                for old_path, path, file_id, ie in delta:
                    if ie is None:
                        continue
                    ie.check(checker, rev_id, inv)
            else:
                for path, ie in inv.iter_entries():
                    ie.check(checker, rev_id, inv)
            if self._format.fast_deltas:
                return inv
        elif kind == 'chk_bytes':
            checker._report_items.append('unsupported key type chk_bytes for {}'.format(record.key))
        elif kind == 'texts':
            self._check_text(record, checker, item_data)
        else:
            checker._report_items.append('unknown key type {} for {}'.format(kind, record.key))

    def _check_text(self, record, checker, item_data):
        """Check a single text."""
        chunks = record.get_bytes_as('chunked')
        sha1 = osutils.sha_strings(chunks)
        length = sum(map(len, chunks))
        if item_data and sha1 != item_data[1]:
            checker._report_items.append('sha1 mismatch: %s has sha1 %s expected %s referenced by %s' % (record.key, sha1, item_data[1], item_data[2]))

    def _eliminate_revisions_not_present(self, revision_ids):
        """Check every revision id in revision_ids to see if we have it.

        Returns a set of the present revisions.
        """
        with self.lock_read():
            result = []
            graph = self.get_graph()
            parent_map = graph.get_parent_map(revision_ids)
            return list(parent_map)

    def __init__(self, _format, a_controldir, control_files):
        """Instantiate a VersionedFileRepository.

        :param _format: The format of the repository on disk.
        :param controldir: The ControlDir of the repository.
        :param control_files: Control files to use for locking, etc.
        """
        super().__init__(_format, a_controldir, control_files)
        self._transport = control_files._transport
        self.base = self._transport.base
        self._reconcile_does_inventory_gc = True
        self._reconcile_fixes_text_parents = False
        self._reconcile_backsup_inventory = True
        self._inventory_entry_cache = fifo_cache.FIFOCache(10 * 1024)
        self._safe_to_return_from_cache = False

    def fetch(self, source, revision_id=None, find_ghosts=False, fetch_spec=None, lossy=False):
        """Fetch the content required to construct revision_id from source.

        If revision_id is None and fetch_spec is None, then all content is
        copied.

        fetch() may not be used when the repository is in a write group -
        either finish the current write group before using fetch, or use
        fetch before starting the write group.

        :param find_ghosts: Find and copy revisions in the source that are
            ghosts in the target (and not reachable directly by walking out to
            the first-present revision in target from revision_id).
        :param revision_id: If specified, all the content needed for this
            revision ID will be copied to the target.  Fetch will determine for
            itself which content needs to be copied.
        :param fetch_spec: If specified, a SearchResult or
            PendingAncestryResult that describes which revisions to copy.  This
            allows copying multiple heads at once.  Mutually exclusive with
            revision_id.
        """
        if fetch_spec is not None and revision_id is not None:
            raise AssertionError('fetch_spec and revision_id are mutually exclusive.')
        if self.is_in_write_group():
            raise errors.InternalBzrError('May not fetch while in a write group.')
        if self.has_same_location(source) and fetch_spec is None and self._has_same_fallbacks(source):
            if revision_id is not None and (not _mod_revision.is_null(revision_id)):
                self.get_revision(revision_id)
            return FetchResult(0)
        inter = InterRepository.get(source, self)
        if fetch_spec is not None and (not getattr(inter, 'supports_fetch_spec', False)):
            raise errors.UnsupportedOperation('fetch_spec not supported for %r' % inter)
        return inter.fetch(revision_id=revision_id, find_ghosts=find_ghosts, fetch_spec=fetch_spec, lossy=lossy)

    def gather_stats(self, revid=None, committers=None):
        """See Repository.gather_stats()."""
        with self.lock_read():
            result = super().gather_stats(revid, committers)
            if self.user_transport.listable():
                result['revisions'] = len(self.revisions.keys())
            return result

    def get_commit_builder(self, branch, parents, config_stack, timestamp=None, timezone=None, committer=None, revprops=None, revision_id=None, lossy=False):
        """Obtain a CommitBuilder for this repository.

        :param branch: Branch to commit to.
        :param parents: Revision ids of the parents of the new revision.
        :param config_stack: Configuration stack to use.
        :param timestamp: Optional timestamp recorded for commit.
        :param timezone: Optional timezone for timestamp.
        :param committer: Optional committer to set for commit.
        :param revprops: Optional dictionary of revision properties.
        :param revision_id: Optional revision id.
        :param lossy: Whether to discard data that can not be natively
            represented, when pushing to a foreign VCS
        """
        if self._fallback_repositories and (not self._format.supports_chks):
            raise errors.BzrError('Cannot commit directly to a stacked branch in pre-2a formats. See https://bugs.launchpad.net/bzr/+bug/375013 for details.')
        in_transaction = self.is_in_write_group()
        result = self._commit_builder_class(self, parents, config_stack, timestamp, timezone, committer, revprops, revision_id, lossy, owns_transaction=not in_transaction)
        if not in_transaction:
            self.start_write_group()
        return result

    def get_missing_parent_inventories(self, check_for_missing_texts=True):
        """Return the keys of missing inventory parents for revisions added in
        this write group.

        A revision is not complete if the inventory delta for that revision
        cannot be calculated.  Therefore if the parent inventories of a
        revision are not present, the revision is incomplete, and e.g. cannot
        be streamed by a smart server.  This method finds missing inventory
        parents for revisions added in this write group.
        """
        if not self._format.supports_external_lookups:
            return set()
        if not self.is_in_write_group():
            raise AssertionError('not in a write group')
        parents = set(self.revisions._index.get_missing_parents())
        parents.discard(_mod_revision.NULL_REVISION)
        unstacked_inventories = self.inventories._index
        present_inventories = unstacked_inventories.get_parent_map((key[-1:] for key in parents))
        parents.difference_update(present_inventories)
        if len(parents) == 0:
            return set()
        if not check_for_missing_texts:
            return {('inventories', rev_id) for rev_id, in parents}
        key_deps = self.revisions._index._key_dependencies
        key_deps.satisfy_refs_for_keys(present_inventories)
        referrers = frozenset((r[0] for r in key_deps.get_referrers()))
        file_ids = self.fileids_altered_by_revision_ids(referrers)
        missing_texts = set()
        for file_id, version_ids in file_ids.items():
            missing_texts.update(((file_id, version_id) for version_id in version_ids))
        present_texts = self.texts.get_parent_map(missing_texts)
        missing_texts.difference_update(present_texts)
        if not missing_texts:
            return set()
        missing_keys = {('inventories', rev_id) for rev_id, in parents}
        return missing_keys

    def has_revisions(self, revision_ids):
        """Probe to find out the presence of multiple revisions.

        :param revision_ids: An iterable of revision_ids.
        :return: A set of the revision_ids that were present.
        """
        with self.lock_read():
            parent_map = self.revisions.get_parent_map([(rev_id,) for rev_id in revision_ids])
            result = set()
            if _mod_revision.NULL_REVISION in revision_ids:
                result.add(_mod_revision.NULL_REVISION)
            result.update([key[0] for key in parent_map])
            return result

    def get_revision_reconcile(self, revision_id):
        """'reconcile' helper routine that allows access to a revision always.

        This variant of get_revision does not cross check the weave graph
        against the revision one as get_revision does: but it should only
        be used by reconcile, or reconcile-alike commands that are correcting
        or testing the revision graph.
        """
        with self.lock_read():
            return self.get_revisions([revision_id])[0]

    def iter_revisions(self, revision_ids):
        """Iterate over revision objects.

        :param revision_ids: An iterable of revisions to examine. None may be
            passed to request all revisions known to the repository. Note that
            not all repositories can find unreferenced revisions; for those
            repositories only referenced ones will be returned.
        :return: An iterator of (revid, revision) tuples. Absent revisions (
            those asked for but not available) are returned as (revid, None).
        """
        with self.lock_read():
            for rev_id in revision_ids:
                if not rev_id or not isinstance(rev_id, bytes):
                    raise errors.InvalidRevisionId(revision_id=rev_id, branch=self)
            keys = [(key,) for key in revision_ids]
            stream = self.revisions.get_record_stream(keys, 'unordered', True)
            for record in stream:
                revid = record.key[0]
                if record.storage_kind == 'absent':
                    yield (revid, None)
                else:
                    text = record.get_bytes_as('fulltext')
                    rev = self._serializer.read_revision_from_string(text)
                    yield (revid, rev)

    def add_signature_text(self, revision_id, signature):
        """Store a signature text for a revision.

        :param revision_id: Revision id of the revision
        :param signature: Signature text.
        """
        with self.lock_write():
            self.signatures.add_lines((revision_id,), (), osutils.split_lines(signature))

    def sign_revision(self, revision_id, gpg_strategy):
        with self.lock_write():
            testament = Testament.from_revision(self, revision_id)
            plaintext = testament.as_short_text()
            self.store_revision_signature(gpg_strategy, plaintext, revision_id)

    def store_revision_signature(self, gpg_strategy, plaintext, revision_id):
        with self.lock_write():
            signature = gpg_strategy.sign(plaintext, gpg.MODE_CLEAR)
            self.add_signature_text(revision_id, signature)

    def verify_revision_signature(self, revision_id, gpg_strategy):
        """Verify the signature on a revision.

        :param revision_id: the revision to verify
        :gpg_strategy: the GPGStrategy object to used

        :return: gpg.SIGNATURE_VALID or a failed SIGNATURE_ value
        """
        with self.lock_read():
            if not self.has_signature_for_revision_id(revision_id):
                return (gpg.SIGNATURE_NOT_SIGNED, None)
            signature = self.get_signature_text(revision_id)
            testament = Testament.from_revision(self, revision_id)
            status, key, signed_plaintext = gpg_strategy.verify(signature)
            if testament.as_short_text() != signed_plaintext:
                return (gpg.SIGNATURE_NOT_VALID, None)
            return (status, key)

    def find_text_key_references(self):
        """Find the text key references within the repository.

        :return: A dictionary mapping text keys ((fileid, revision_id) tuples)
            to whether they were referred to by the inventory of the
            revision_id that they contain. The inventory texts from all present
            revision ids are assessed to generate this report.
        """
        revision_keys = self.revisions.keys()
        w = self.inventories
        with ui.ui_factory.nested_progress_bar() as pb:
            return self._serializer._find_text_key_references(w.iter_lines_added_or_present_in_keys(revision_keys, pb=pb))

    def _inventory_xml_lines_for_keys(self, keys):
        """Get a line iterator of the sort needed for findind references.

        Not relevant for non-xml inventory repositories.

        Ghosts in revision_keys are ignored.

        :param revision_keys: The revision keys for the inventories to inspect.
        :return: An iterator over (inventory line, revid) for the fulltexts of
            all of the xml inventories specified by revision_keys.
        """
        stream = self.inventories.get_record_stream(keys, 'unordered', True)
        for record in stream:
            if record.storage_kind != 'absent':
                revid = record.key[-1]
                for line in record.get_bytes_as('lines'):
                    yield (line, revid)

    def _find_file_ids_from_xml_inventory_lines(self, line_iterator, revision_keys):
        """Helper routine for fileids_altered_by_revision_ids.

        This performs the translation of xml lines to revision ids.

        :param line_iterator: An iterator of lines, origin_version_id
        :param revision_keys: The revision ids to filter for. This should be a
            set or other type which supports efficient __contains__ lookups, as
            the revision key from each parsed line will be looked up in the
            revision_keys filter.
        :return: a dictionary mapping altered file-ids to an iterable of
            revision_ids. Each altered file-ids has the exact revision_ids that
            altered it listed explicitly.
        """
        seen = set(self._serializer._find_text_key_references(line_iterator))
        parent_keys = self._find_parent_keys_of_revisions(revision_keys)
        parent_seen = set(self._serializer._find_text_key_references(self._inventory_xml_lines_for_keys(parent_keys)))
        new_keys = seen - parent_seen
        result = {}
        setdefault = result.setdefault
        for key in new_keys:
            setdefault(key[0], set()).add(key[-1])
        return result

    def _find_parent_keys_of_revisions(self, revision_keys):
        """Similar to _find_parent_ids_of_revisions, but used with keys.

        :param revision_keys: An iterable of revision_keys.
        :return: The parents of all revision_keys that are not already in
            revision_keys
        """
        parent_map = self.revisions.get_parent_map(revision_keys)
        parent_keys = set(itertools.chain.from_iterable(parent_map.values()))
        parent_keys.difference_update(revision_keys)
        parent_keys.discard(_mod_revision.NULL_REVISION)
        return parent_keys

    def fileids_altered_by_revision_ids(self, revision_ids, _inv_weave=None):
        """Find the file ids and versions affected by revisions.

        :param revisions: an iterable containing revision ids.
        :param _inv_weave: The inventory weave from this repository or None.
            If None, the inventory weave will be opened automatically.
        :return: a dictionary mapping altered file-ids to an iterable of
            revision_ids. Each altered file-ids has the exact revision_ids that
            altered it listed explicitly.
        """
        selected_keys = {(revid,) for revid in revision_ids}
        w = _inv_weave or self.inventories
        return self._find_file_ids_from_xml_inventory_lines(w.iter_lines_added_or_present_in_keys(selected_keys, pb=None), selected_keys)

    def iter_files_bytes(self, desired_files):
        """Iterate through file versions.

        Files will not necessarily be returned in the order they occur in
        desired_files.  No specific order is guaranteed.

        Yields pairs of identifier, bytes_iterator.  identifier is an opaque
        value supplied by the caller as part of desired_files.  It should
        uniquely identify the file version in the caller's context.  (Examples:
        an index number or a TreeTransform trans_id.)

        bytes_iterator is an iterable of bytestrings for the file.  The
        kind of iterable and length of the bytestrings are unspecified, but for
        this implementation, it is a list of bytes produced by
        VersionedFile.get_record_stream().

        :param desired_files: a list of (file_id, revision_id, identifier)
            triples
        """
        text_keys = {}
        for file_id, revision_id, callable_data in desired_files:
            text_keys[file_id, revision_id] = callable_data
        for record in self.texts.get_record_stream(text_keys, 'unordered', True):
            if record.storage_kind == 'absent':
                raise errors.RevisionNotPresent(record.key[1], record.key[0])
            yield (text_keys[record.key], record.iter_bytes_as('chunked'))

    def _generate_text_key_index(self, text_key_references=None, ancestors=None):
        """Generate a new text key index for the repository.

        This is an expensive function that will take considerable time to run.

        :return: A dict mapping text keys ((file_id, revision_id) tuples) to a
            list of parents, also text keys. When a given key has no parents,
            the parents list will be [NULL_REVISION].
        """
        if ancestors is None:
            graph = self.get_graph()
            ancestors = graph.get_parent_map(self.all_revision_ids())
        if text_key_references is None:
            text_key_references = self.find_text_key_references()
        with ui.ui_factory.nested_progress_bar() as pb:
            return self._do_generate_text_key_index(ancestors, text_key_references, pb)

    def _do_generate_text_key_index(self, ancestors, text_key_references, pb):
        """Helper for _generate_text_key_index to avoid deep nesting."""
        revision_order = tsort.topo_sort(ancestors)
        invalid_keys = set()
        revision_keys = {}
        for revision_id in revision_order:
            revision_keys[revision_id] = set()
        text_count = len(text_key_references)
        text_key_cache = {}
        for text_key, valid in text_key_references.items():
            if not valid:
                invalid_keys.add(text_key)
            else:
                revision_keys[text_key[1]].add(text_key)
            text_key_cache[text_key] = text_key
        del text_key_references
        text_index = {}
        text_graph = graph.Graph(graph.DictParentsProvider(text_index))
        NULL_REVISION = _mod_revision.NULL_REVISION
        inventory_cache = lru_cache.LRUCache(10)
        batch_size = 10
        batch_count = len(revision_order) // batch_size + 1
        processed_texts = 0
        pb.update(gettext('Calculating text parents'), processed_texts, text_count)
        for offset in range(batch_count):
            to_query = revision_order[offset * batch_size:(offset + 1) * batch_size]
            if not to_query:
                break
            for revision_id in to_query:
                parent_ids = ancestors[revision_id]
                for text_key in revision_keys[revision_id]:
                    pb.update(gettext('Calculating text parents'), processed_texts)
                    processed_texts += 1
                    candidate_parents = []
                    for parent_id in parent_ids:
                        parent_text_key = (text_key[0], parent_id)
                        try:
                            check_parent = parent_text_key not in revision_keys[parent_id]
                        except KeyError:
                            check_parent = False
                            parent_text_key = None
                        if check_parent:
                            try:
                                inv = inventory_cache[parent_id]
                            except KeyError:
                                inv = self.revision_tree(parent_id).root_inventory
                                inventory_cache[parent_id] = inv
                            try:
                                parent_entry = inv.get_entry(text_key[0])
                            except (KeyError, errors.NoSuchId):
                                parent_entry = None
                            if parent_entry is not None:
                                parent_text_key = (text_key[0], parent_entry.revision)
                            else:
                                parent_text_key = None
                        if parent_text_key is not None:
                            candidate_parents.append(text_key_cache[parent_text_key])
                    parent_heads = text_graph.heads(candidate_parents)
                    new_parents = list(parent_heads)
                    new_parents.sort(key=lambda x: candidate_parents.index(x))
                    if new_parents == []:
                        new_parents = [NULL_REVISION]
                    text_index[text_key] = new_parents
        for text_key in invalid_keys:
            text_index[text_key] = [NULL_REVISION]
        return text_index

    def item_keys_introduced_by(self, revision_ids, _files_pb=None):
        """Get an iterable listing the keys of all the data introduced by a set
        of revision IDs.

        The keys will be ordered so that the corresponding items can be safely
        fetched and inserted in that order.

        :returns: An iterable producing tuples of (knit-kind, file-id,
            versions).  knit-kind is one of 'file', 'inventory', 'signatures',
            'revisions'.  file-id is None unless knit-kind is 'file'.
        """
        yield from self._find_file_keys_to_fetch(revision_ids, _files_pb)
        del _files_pb
        yield from self._find_non_file_keys_to_fetch(revision_ids)

    def _find_file_keys_to_fetch(self, revision_ids, pb):
        inv_w = self.inventories
        file_ids = self.fileids_altered_by_revision_ids(revision_ids, inv_w)
        count = 0
        num_file_ids = len(file_ids)
        for file_id, altered_versions in file_ids.items():
            if pb is not None:
                pb.update(gettext('Fetch texts'), count, num_file_ids)
            count += 1
            yield ('file', file_id, altered_versions)

    def _find_non_file_keys_to_fetch(self, revision_ids):
        yield ('inventory', None, revision_ids)
        revisions_with_signatures = set(self.signatures.get_parent_map([(r,) for r in revision_ids]))
        revisions_with_signatures = {r for r, in revisions_with_signatures}
        revisions_with_signatures.intersection_update(revision_ids)
        yield ('signatures', None, revisions_with_signatures)
        yield ('revisions', None, revision_ids)

    def get_inventory(self, revision_id):
        """Get Inventory object by revision id."""
        with self.lock_read():
            return next(self.iter_inventories([revision_id]))

    def iter_inventories(self, revision_ids, ordering=None):
        """Get many inventories by revision_ids.

        This will buffer some or all of the texts used in constructing the
        inventories in memory, but will only parse a single inventory at a
        time.

        :param revision_ids: The expected revision ids of the inventories.
        :param ordering: optional ordering, e.g. 'topological'.  If not
            specified, the order of revision_ids will be preserved (by
            buffering if necessary).
        :return: An iterator of inventories.
        """
        if None in revision_ids or _mod_revision.NULL_REVISION in revision_ids:
            raise ValueError('cannot get null revision inventory')
        for inv, revid in self._iter_inventories(revision_ids, ordering):
            if inv is None:
                raise errors.NoSuchRevision(self, revid)
            yield inv

    def _iter_inventories(self, revision_ids, ordering):
        """single-document based inventory iteration."""
        inv_xmls = self._iter_inventory_xmls(revision_ids, ordering)
        for lines, revision_id in inv_xmls:
            if lines is None:
                yield (None, revision_id)
            else:
                yield (self._deserialise_inventory(revision_id, lines), revision_id)

    def _iter_inventory_xmls(self, revision_ids, ordering):
        if ordering is None:
            order_as_requested = True
            ordering = 'unordered'
        else:
            order_as_requested = False
        keys = [(revision_id,) for revision_id in revision_ids]
        if not keys:
            return
        if order_as_requested:
            key_iter = iter(keys)
            next_key = next(key_iter)
        stream = self.inventories.get_record_stream(keys, ordering, True)
        text_lines = {}
        for record in stream:
            if record.storage_kind != 'absent':
                lines = record.get_bytes_as('lines')
                if order_as_requested:
                    text_lines[record.key] = lines
                else:
                    yield (lines, record.key[-1])
            else:
                yield (None, record.key[-1])
            if order_as_requested:
                while next_key in text_lines:
                    lines = text_lines.pop(next_key)
                    yield (lines, next_key[-1])
                    try:
                        next_key = next(key_iter)
                    except StopIteration:
                        next_key = None
                        break

    def _deserialise_inventory(self, revision_id, xml):
        """Transform the xml into an inventory object.

        :param revision_id: The expected revision id of the inventory.
        :param xml: A serialised inventory.
        """
        result = self._serializer.read_inventory_from_lines(xml, revision_id, entry_cache=self._inventory_entry_cache, return_from_cache=self._safe_to_return_from_cache)
        if result.revision_id != revision_id:
            raise AssertionError('revision id mismatch {} != {}'.format(result.revision_id, revision_id))
        return result

    def get_serializer_format(self):
        return self._serializer.format_num

    def _get_inventory_xml(self, revision_id):
        """Get serialized inventory as a string."""
        with self.lock_read():
            texts = self._iter_inventory_xmls([revision_id], 'unordered')
            lines, revision_id = next(texts)
            if lines is None:
                raise errors.NoSuchRevision(self, revision_id)
            return lines

    def revision_tree(self, revision_id):
        """Return Tree for a revision on this branch.

        `revision_id` may be NULL_REVISION for the empty tree revision.
        """
        if revision_id == _mod_revision.NULL_REVISION:
            return inventorytree.InventoryRevisionTree(self, Inventory(root_id=None), _mod_revision.NULL_REVISION)
        else:
            with self.lock_read():
                inv = self.get_inventory(revision_id)
                return inventorytree.InventoryRevisionTree(self, inv, revision_id)

    def revision_trees(self, revision_ids):
        """Return Trees for revisions in this repository.

        :param revision_ids: a sequence of revision-ids;
          a revision-id may not be None or b'null:'
        """
        inventories = self.iter_inventories(revision_ids)
        for inv in inventories:
            yield inventorytree.InventoryRevisionTree(self, inv, inv.revision_id)

    def get_parent_map(self, revision_ids):
        """See graph.StackedParentsProvider.get_parent_map"""
        query_keys = []
        result = {}
        for revision_id in revision_ids:
            if revision_id == _mod_revision.NULL_REVISION:
                result[revision_id] = ()
            elif revision_id is None:
                raise ValueError('get_parent_map(None) is not valid')
            else:
                query_keys.append((revision_id,))
        for (revision_id,), parent_keys in self.revisions.get_parent_map(query_keys).items():
            if parent_keys:
                result[revision_id] = tuple([parent_revid for parent_revid, in parent_keys])
            else:
                result[revision_id] = (_mod_revision.NULL_REVISION,)
        return result

    def get_known_graph_ancestry(self, revision_ids):
        """Return the known graph for a set of revision ids and their ancestors.
        """
        st = static_tuple.StaticTuple
        revision_keys = [st(r_id).intern() for r_id in revision_ids]
        with self.lock_read():
            known_graph = self.revisions.get_known_graph_ancestry(revision_keys)
            return graph.GraphThunkIdsToKeys(known_graph)

    def get_file_graph(self):
        """Return the graph walker for text revisions."""
        with self.lock_read():
            return graph.Graph(self.texts)

    def revision_ids_to_search_result(self, result_set):
        """Convert a set of revision ids to a graph SearchResult."""
        result_parents = set(itertools.chain.from_iterable(self.get_graph().get_parent_map(result_set).values()))
        included_keys = result_set.intersection(result_parents)
        start_keys = result_set.difference(included_keys)
        exclude_keys = result_parents.difference(result_set)
        result = vf_search.SearchResult(start_keys, exclude_keys, len(result_set), result_set)
        return result

    def _get_versioned_file_checker(self, text_key_references=None, ancestors=None):
        """Return an object suitable for checking versioned files.

        :param text_key_references: if non-None, an already built
            dictionary mapping text keys ((fileid, revision_id) tuples)
            to whether they were referred to by the inventory of the
            revision_id that they contain. If None, this will be
            calculated.
        :param ancestors: Optional result from
            self.get_graph().get_parent_map(self.all_revision_ids()) if already
            available.
        """
        return _VersionedFileChecker(self, text_key_references=text_key_references, ancestors=ancestors)

    def has_signature_for_revision_id(self, revision_id):
        """Query for a revision signature for revision_id in the repository."""
        with self.lock_read():
            if not self.has_revision(revision_id):
                raise errors.NoSuchRevision(self, revision_id)
            sig_present = 1 == len(self.signatures.get_parent_map([(revision_id,)]))
            return sig_present

    def get_signature_text(self, revision_id):
        """Return the text for a signature."""
        with self.lock_read():
            stream = self.signatures.get_record_stream([(revision_id,)], 'unordered', True)
            record = next(stream)
            if record.storage_kind == 'absent':
                raise errors.NoSuchRevision(self, revision_id)
            return record.get_bytes_as('fulltext')

    def _check(self, revision_ids, callback_refs, check_repo):
        with self.lock_read():
            result = check.VersionedFileCheck(self, check_repo=check_repo)
            result.check(callback_refs)
            return result

    def _find_inconsistent_revision_parents(self, revisions_iterator=None):
        """Find revisions with different parent lists in the revision object
        and in the index graph.

        :param revisions_iterator: None, or an iterator of (revid,
            Revision-or-None). This iterator controls the revisions checked.
        :returns: an iterator yielding tuples of (revison-id, parents-in-index,
            parents-in-revision).
        """
        if not self.is_locked():
            raise AssertionError()
        vf = self.revisions
        if revisions_iterator is None:
            revisions_iterator = self.iter_revisions(self.all_revision_ids())
        for revid, revision in revisions_iterator:
            if revision is None:
                pass
            parent_map = vf.get_parent_map([(revid,)])
            parents_according_to_index = tuple((parent[-1] for parent in parent_map[revid,]))
            parents_according_to_revision = tuple(revision.parent_ids)
            if parents_according_to_index != parents_according_to_revision:
                yield (revid, parents_according_to_index, parents_according_to_revision)

    def _check_for_inconsistent_revision_parents(self):
        inconsistencies = list(self._find_inconsistent_revision_parents())
        if inconsistencies:
            raise errors.BzrCheckError('Revision knit has inconsistent parents.')

    def _get_sink(self):
        """Return a sink for streaming into this repository."""
        return StreamSink(self)

    def _get_source(self, to_format):
        """Return a source for streaming from this repository."""
        return StreamSource(self, to_format)

    def reconcile(self, other=None, thorough=False):
        """Reconcile this repository."""
        from .reconcile import VersionedFileRepoReconciler
        with self.lock_write():
            reconciler = VersionedFileRepoReconciler(self, thorough=thorough)
            return reconciler.reconcile()