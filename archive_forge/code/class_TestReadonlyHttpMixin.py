import itertools
from gzip import GzipFile
from io import BytesIO
from ... import errors
from ... import graph as _mod_graph
from ... import osutils, progress, transport, ui
from ...errors import RevisionAlreadyPresent, RevisionNotPresent
from ...tests import (TestCase, TestCaseWithMemoryTransport, TestNotApplicable,
from ...tests.http_utils import TestCaseWithWebserver
from ...tests.scenarios import load_tests_apply_scenarios
from ...transport.memory import MemoryTransport
from .. import groupcompress
from .. import knit as _mod_knit
from .. import versionedfile as versionedfile
from ..knit import cleanup_pack_knit, make_file_factory, make_pack_factory
from ..versionedfile import (ChunkedContentFactory, ConstantMapper,
from ..weave import WeaveFile, WeaveInvalidChecksum
from ..weavefile import write_weave
class TestReadonlyHttpMixin:

    def get_transaction(self):
        return 1

    def test_readonly_http_works(self):
        vf = self.get_file()
        readonly_vf = self.get_factory()('foo', transport.get_transport_from_url(self.get_readonly_url('.')))
        self.assertEqual([], readonly_vf.versions())

    def test_readonly_http_works_with_feeling(self):
        vf = self.get_file()
        vf.add_lines(b'1', [], [b'a\n'])
        vf.add_lines(b'2', [b'1'], [b'b\n', b'a\n'])
        readonly_vf = self.get_factory()('foo', transport.get_transport_from_url(self.get_readonly_url('.')))
        self.assertEqual([b'1', b'2'], vf.versions())
        self.assertEqual([b'1', b'2'], readonly_vf.versions())
        for version in readonly_vf.versions():
            readonly_vf.get_lines(version)