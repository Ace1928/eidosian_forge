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
class TestContentFactoryAdaption(TestCaseWithMemoryTransport):

    def test_select_adaptor(self):
        """Test expected adapters exist."""
        scenarios = [('knit-delta-gz', 'fulltext', _mod_knit.DeltaPlainToFullText), ('knit-delta-gz', 'lines', _mod_knit.DeltaPlainToFullText), ('knit-delta-gz', 'chunked', _mod_knit.DeltaPlainToFullText), ('knit-ft-gz', 'fulltext', _mod_knit.FTPlainToFullText), ('knit-ft-gz', 'lines', _mod_knit.FTPlainToFullText), ('knit-ft-gz', 'chunked', _mod_knit.FTPlainToFullText), ('knit-annotated-delta-gz', 'knit-delta-gz', _mod_knit.DeltaAnnotatedToUnannotated), ('knit-annotated-delta-gz', 'fulltext', _mod_knit.DeltaAnnotatedToFullText), ('knit-annotated-ft-gz', 'knit-ft-gz', _mod_knit.FTAnnotatedToUnannotated), ('knit-annotated-ft-gz', 'fulltext', _mod_knit.FTAnnotatedToFullText), ('knit-annotated-ft-gz', 'lines', _mod_knit.FTAnnotatedToFullText), ('knit-annotated-ft-gz', 'chunked', _mod_knit.FTAnnotatedToFullText)]
        for source, requested, klass in scenarios:
            adapter_factory = versionedfile.adapter_registry.get((source, requested))
            adapter = adapter_factory(None)
            self.assertIsInstance(adapter, klass)

    def get_knit(self, annotated=True):
        mapper = ConstantMapper('knit')
        transport = self.get_transport()
        return make_file_factory(annotated, mapper)(transport)

    def helpGetBytes(self, f, ft_name, ft_adapter, delta_name, delta_adapter):
        """Grab the interested adapted texts for tests."""
        entries = f.get_record_stream([(b'origin',)], 'unordered', False)
        base = next(entries)
        ft_data = ft_adapter.get_bytes(base, ft_name)
        entries = f.get_record_stream([(b'merged',)], 'unordered', False)
        merged = next(entries)
        delta_data = delta_adapter.get_bytes(merged, delta_name)
        return (ft_data, delta_data)

    def test_deannotation_noeol(self):
        """Test converting annotated knits to unannotated knits."""
        f = self.get_knit()
        get_diamond_files(f, 1, trailing_eol=False)
        ft_data, delta_data = self.helpGetBytes(f, 'knit-ft-gz', _mod_knit.FTAnnotatedToUnannotated(None), 'knit-delta-gz', _mod_knit.DeltaAnnotatedToUnannotated(None))
        self.assertEqual(b'version origin 1 b284f94827db1fa2970d9e2014f080413b547a7e\norigin\nend origin\n', GzipFile(mode='rb', fileobj=BytesIO(ft_data)).read())
        self.assertEqual(b'version merged 4 32c2e79763b3f90e8ccde37f9710b6629c25a796\n1,2,3\nleft\nright\nmerged\nend merged\n', GzipFile(mode='rb', fileobj=BytesIO(delta_data)).read())

    def test_deannotation(self):
        """Test converting annotated knits to unannotated knits."""
        f = self.get_knit()
        get_diamond_files(f, 1)
        ft_data, delta_data = self.helpGetBytes(f, 'knit-ft-gz', _mod_knit.FTAnnotatedToUnannotated(None), 'knit-delta-gz', _mod_knit.DeltaAnnotatedToUnannotated(None))
        self.assertEqual(b'version origin 1 00e364d235126be43292ab09cb4686cf703ddc17\norigin\nend origin\n', GzipFile(mode='rb', fileobj=BytesIO(ft_data)).read())
        self.assertEqual(b'version merged 3 ed8bce375198ea62444dc71952b22cfc2b09226d\n2,2,2\nright\nmerged\nend merged\n', GzipFile(mode='rb', fileobj=BytesIO(delta_data)).read())

    def test_annotated_to_fulltext_no_eol(self):
        """Test adapting annotated knits to full texts (for -> weaves)."""
        f = self.get_knit()
        get_diamond_files(f, 1, trailing_eol=False)
        logged_vf = versionedfile.RecordingVersionedFilesDecorator(f)
        ft_data, delta_data = self.helpGetBytes(f, 'fulltext', _mod_knit.FTAnnotatedToFullText(None), 'fulltext', _mod_knit.DeltaAnnotatedToFullText(logged_vf))
        self.assertEqual(b'origin', ft_data)
        self.assertEqual(b'base\nleft\nright\nmerged', delta_data)
        self.assertEqual([('get_record_stream', [(b'left',)], 'unordered', True)], logged_vf.calls)

    def test_annotated_to_fulltext(self):
        """Test adapting annotated knits to full texts (for -> weaves)."""
        f = self.get_knit()
        get_diamond_files(f, 1)
        logged_vf = versionedfile.RecordingVersionedFilesDecorator(f)
        ft_data, delta_data = self.helpGetBytes(f, 'fulltext', _mod_knit.FTAnnotatedToFullText(None), 'fulltext', _mod_knit.DeltaAnnotatedToFullText(logged_vf))
        self.assertEqual(b'origin\n', ft_data)
        self.assertEqual(b'base\nleft\nright\nmerged\n', delta_data)
        self.assertEqual([('get_record_stream', [(b'left',)], 'unordered', True)], logged_vf.calls)

    def test_unannotated_to_fulltext(self):
        """Test adapting unannotated knits to full texts.

        This is used for -> weaves, and for -> annotated knits.
        """
        f = self.get_knit(annotated=False)
        get_diamond_files(f, 1)
        logged_vf = versionedfile.RecordingVersionedFilesDecorator(f)
        ft_data, delta_data = self.helpGetBytes(f, 'fulltext', _mod_knit.FTPlainToFullText(None), 'fulltext', _mod_knit.DeltaPlainToFullText(logged_vf))
        self.assertEqual(b'origin\n', ft_data)
        self.assertEqual(b'base\nleft\nright\nmerged\n', delta_data)
        self.assertEqual([('get_record_stream', [(b'left',)], 'unordered', True)], logged_vf.calls)

    def test_unannotated_to_fulltext_no_eol(self):
        """Test adapting unannotated knits to full texts.

        This is used for -> weaves, and for -> annotated knits.
        """
        f = self.get_knit(annotated=False)
        get_diamond_files(f, 1, trailing_eol=False)
        logged_vf = versionedfile.RecordingVersionedFilesDecorator(f)
        ft_data, delta_data = self.helpGetBytes(f, 'fulltext', _mod_knit.FTPlainToFullText(None), 'fulltext', _mod_knit.DeltaPlainToFullText(logged_vf))
        self.assertEqual(b'origin', ft_data)
        self.assertEqual(b'base\nleft\nright\nmerged', delta_data)
        self.assertEqual([('get_record_stream', [(b'left',)], 'unordered', True)], logged_vf.calls)