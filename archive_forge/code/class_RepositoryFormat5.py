import gzip
import os
from io import BytesIO
from ...lazy_import import lazy_import
import itertools
from breezy import (
from breezy.bzr import (
from ... import debug, errors, lockable_files, lockdir, osutils, trace
from ... import transport as _mod_transport
from ... import urlutils
from ...bzr import tuned_gzip, versionedfile, weave, weavefile
from ...bzr.repository import RepositoryFormatMetaDir
from ...bzr.versionedfile import (AbsentContentFactory, FulltextContentFactory,
from ...bzr.vf_repository import (InterSameDataRepository,
from ...repository import InterRepository
from . import bzrdir as weave_bzrdir
from .store.text import TextStore
class RepositoryFormat5(PreSplitOutRepositoryFormat):
    """Bzr control format 5.

    This repository format has:
     - weaves for file texts and inventory
     - flat stores
     - TextStores for revisions and signatures.
    """
    _versionedfile_class = weave.WeaveFile
    _matchingcontroldir = weave_bzrdir.BzrDirFormat5()
    supports_funky_characters = False

    @property
    def _serializer(self):
        return xml5.serializer_v5

    def get_format_description(self):
        """See RepositoryFormat.get_format_description()."""
        return 'Weave repository format 5'

    def network_name(self):
        """The network name for this format is the control dirs disk label."""
        return self._matchingcontroldir.get_format_string()

    def _get_inventories(self, repo_transport, repo, name='inventory'):
        mapper = versionedfile.ConstantMapper(name)
        return versionedfile.ThunkedVersionedFiles(repo_transport, weave.WeaveFile, mapper, repo.is_locked)

    def _get_revisions(self, repo_transport, repo):
        return RevisionTextStore(repo_transport.clone('revision-store'), xml5.serializer_v5, False, versionedfile.PrefixMapper(), repo.is_locked, repo.is_write_locked)

    def _get_signatures(self, repo_transport, repo):
        return SignatureTextStore(repo_transport.clone('revision-store'), False, versionedfile.PrefixMapper(), repo.is_locked, repo.is_write_locked)

    def _get_texts(self, repo_transport, repo):
        mapper = versionedfile.PrefixMapper()
        base_transport = repo_transport.clone('weaves')
        return versionedfile.ThunkedVersionedFiles(base_transport, weave.WeaveFile, mapper, repo.is_locked)