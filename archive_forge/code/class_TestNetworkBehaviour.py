import gzip
import sys
from io import BytesIO
from patiencediff import PatienceSequenceMatcher
from ... import errors, multiparent, osutils, tests
from ... import transport as _mod_transport
from ...tests import (TestCase, TestCaseWithMemoryTransport,
from .. import knit, knitpack_repo, pack, pack_repo
from ..index import *
from ..knit import (AnnotatedKnitContent, KnitContent, KnitCorrupt,
from ..versionedfile import (AbsentContentFactory, ConstantMapper,
class TestNetworkBehaviour(KnitTests):
    """Tests for getting data out of/into knits over the network."""

    def test_include_delta_closure_generates_a_knit_delta_closure(self):
        vf = self.make_test_knit(name='test')
        vf.add_lines((b'base',), (), [b'base\n', b'content\n'])
        vf.add_lines((b'd1',), ((b'base',),), [b'd1\n'])
        vf.add_lines((b'd2',), ((b'd1',),), [b'd2\n'])
        self.assertEqual(['knit-ft-gz', 'knit-delta-gz', 'knit-delta-gz'], [record.storage_kind for record in vf.get_record_stream([(b'base',), (b'd1',), (b'd2',)], 'topological', False)])
        stream = vf.get_record_stream([(b'd1',), (b'd2',)], 'topological', True)
        netb = [record.get_bytes_as(record.storage_kind) for record in stream]
        self.assertEqual(b'', netb[1])
        bytes = netb[0]
        kind, line_end = network_bytes_to_kind_and_offset(bytes)
        self.assertEqual('knit-delta-closure', kind)