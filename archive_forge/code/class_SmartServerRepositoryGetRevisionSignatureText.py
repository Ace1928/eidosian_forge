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
class SmartServerRepositoryGetRevisionSignatureText(SmartServerRepositoryRequest):
    """Return the signature text of a revision.

    New in 2.5.
    """

    def do_repository_request(self, repository, revision_id):
        """Return the result of repository.get_signature_text().

        :param repository: The repository to query in.
        :return: A smart server response of with the signature text as
            body.
        """
        try:
            text = repository.get_signature_text(revision_id)
        except errors.NoSuchRevision as err:
            return FailedSmartServerResponse((b'nosuchrevision', err.revision))
        return SuccessfulSmartServerResponse((b'ok',), text)