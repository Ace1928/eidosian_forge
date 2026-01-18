import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
class TestLazyGroupCompress(tests.TestCaseWithTransport):
    _texts = {(b'key1',): b'this is a text\nwith a reasonable amount of compressible bytes\nwhich can be shared between various other texts\n', (b'key2',): b'another text\nwith a reasonable amount of compressible bytes\nwhich can be shared between various other texts\n', (b'key3',): b"yet another text which won't be extracted\nwith a reasonable amount of compressible bytes\nwhich can be shared between various other texts\n", (b'key4',): b"this will be extracted\nbut references most of its bytes from\nyet another text which won't be extracted\nwith a reasonable amount of compressible bytes\nwhich can be shared between various other texts\n"}

    def make_block(self, key_to_text):
        """Create a GroupCompressBlock, filling it with the given texts."""
        compressor = groupcompress.GroupCompressor()
        start = 0
        for key in sorted(key_to_text):
            compressor.compress(key, [key_to_text[key]], len(key_to_text[key]), None)
        locs = {key: (start, end) for key, (start, _, end, _) in compressor.labels_deltas.items()}
        block = compressor.flush()
        raw_bytes = block.to_bytes()
        return (locs, groupcompress.GroupCompressBlock.from_bytes(raw_bytes))

    def add_key_to_manager(self, key, locations, block, manager):
        start, end = locations[key]
        manager.add_factory(key, (), start, end)

    def make_block_and_full_manager(self, texts):
        locations, block = self.make_block(texts)
        manager = groupcompress._LazyGroupContentManager(block)
        for key in sorted(texts):
            self.add_key_to_manager(key, locations, block, manager)
        return (block, manager)

    def test_get_fulltexts(self):
        locations, block = self.make_block(self._texts)
        manager = groupcompress._LazyGroupContentManager(block)
        self.add_key_to_manager((b'key1',), locations, block, manager)
        self.add_key_to_manager((b'key2',), locations, block, manager)
        result_order = []
        for record in manager.get_record_stream():
            result_order.append(record.key)
            text = self._texts[record.key]
            self.assertEqual(text, record.get_bytes_as('fulltext'))
        self.assertEqual([(b'key1',), (b'key2',)], result_order)
        manager = groupcompress._LazyGroupContentManager(block)
        self.add_key_to_manager((b'key2',), locations, block, manager)
        self.add_key_to_manager((b'key1',), locations, block, manager)
        result_order = []
        for record in manager.get_record_stream():
            result_order.append(record.key)
            text = self._texts[record.key]
            self.assertEqual(text, record.get_bytes_as('fulltext'))
        self.assertEqual([(b'key2',), (b'key1',)], result_order)

    def test__wire_bytes_no_keys(self):
        locations, block = self.make_block(self._texts)
        manager = groupcompress._LazyGroupContentManager(block)
        wire_bytes = manager._wire_bytes()
        block_length = len(block.to_bytes())
        stripped_block = manager._block.to_bytes()
        self.assertTrue(block_length > len(stripped_block))
        empty_z_header = zlib.compress(b'')
        self.assertEqual(b'groupcompress-block\n8\n0\n%d\n%s%s' % (len(stripped_block), empty_z_header, stripped_block), wire_bytes)

    def test__wire_bytes(self):
        locations, block = self.make_block(self._texts)
        manager = groupcompress._LazyGroupContentManager(block)
        self.add_key_to_manager((b'key1',), locations, block, manager)
        self.add_key_to_manager((b'key4',), locations, block, manager)
        block_bytes = block.to_bytes()
        wire_bytes = manager._wire_bytes()
        storage_kind, z_header_len, header_len, block_len, rest = wire_bytes.split(b'\n', 4)
        z_header_len = int(z_header_len)
        header_len = int(header_len)
        block_len = int(block_len)
        self.assertEqual(b'groupcompress-block', storage_kind)
        self.assertEqual(34, z_header_len)
        self.assertEqual(26, header_len)
        self.assertEqual(len(block_bytes), block_len)
        z_header = rest[:z_header_len]
        header = zlib.decompress(z_header)
        self.assertEqual(header_len, len(header))
        entry1 = locations[b'key1',]
        entry4 = locations[b'key4',]
        self.assertEqualDiff(b'key1\n\n%d\n%d\nkey4\n\n%d\n%d\n' % (entry1[0], entry1[1], entry4[0], entry4[1]), header)
        z_block = rest[z_header_len:]
        self.assertEqual(block_bytes, z_block)

    def test_from_bytes(self):
        locations, block = self.make_block(self._texts)
        manager = groupcompress._LazyGroupContentManager(block)
        self.add_key_to_manager((b'key1',), locations, block, manager)
        self.add_key_to_manager((b'key4',), locations, block, manager)
        wire_bytes = manager._wire_bytes()
        self.assertStartsWith(wire_bytes, b'groupcompress-block\n')
        manager = groupcompress._LazyGroupContentManager.from_bytes(wire_bytes)
        self.assertIsInstance(manager, groupcompress._LazyGroupContentManager)
        self.assertEqual(2, len(manager._factories))
        self.assertEqual(block._z_content, manager._block._z_content)
        result_order = []
        for record in manager.get_record_stream():
            result_order.append(record.key)
            text = self._texts[record.key]
            self.assertEqual(text, record.get_bytes_as('fulltext'))
        self.assertEqual([(b'key1',), (b'key4',)], result_order)

    def test__check_rebuild_no_changes(self):
        block, manager = self.make_block_and_full_manager(self._texts)
        manager._check_rebuild_block()
        self.assertIs(block, manager._block)

    def test__check_rebuild_only_one(self):
        locations, block = self.make_block(self._texts)
        manager = groupcompress._LazyGroupContentManager(block)
        self.add_key_to_manager((b'key1',), locations, block, manager)
        manager._check_rebuild_block()
        self.assertIsNot(block, manager._block)
        self.assertTrue(block._content_length > manager._block._content_length)
        for record in manager.get_record_stream():
            self.assertEqual((b'key1',), record.key)
            self.assertEqual(self._texts[record.key], record.get_bytes_as('fulltext'))

    def test__check_rebuild_middle(self):
        locations, block = self.make_block(self._texts)
        manager = groupcompress._LazyGroupContentManager(block)
        self.add_key_to_manager((b'key4',), locations, block, manager)
        manager._check_rebuild_block()
        self.assertIsNot(block, manager._block)
        self.assertTrue(block._content_length > manager._block._content_length)
        for record in manager.get_record_stream():
            self.assertEqual((b'key4',), record.key)
            self.assertEqual(self._texts[record.key], record.get_bytes_as('fulltext'))

    def test_manager_default_compressor_settings(self):
        locations, old_block = self.make_block(self._texts)
        manager = groupcompress._LazyGroupContentManager(old_block)
        gcvf = groupcompress.GroupCompressVersionedFiles
        self.assertIs(None, manager._compressor_settings)
        self.assertEqual(gcvf._DEFAULT_COMPRESSOR_SETTINGS, manager._get_compressor_settings())

    def test_manager_custom_compressor_settings(self):
        locations, old_block = self.make_block(self._texts)
        called = []

        def compressor_settings():
            called.append('called')
            return (10,)
        manager = groupcompress._LazyGroupContentManager(old_block, get_compressor_settings=compressor_settings)
        gcvf = groupcompress.GroupCompressVersionedFiles
        self.assertIs(None, manager._compressor_settings)
        self.assertEqual((10,), manager._get_compressor_settings())
        self.assertEqual((10,), manager._get_compressor_settings())
        self.assertEqual((10,), manager._compressor_settings)
        self.assertEqual(['called'], called)

    def test__rebuild_handles_compressor_settings(self):
        if not isinstance(groupcompress.GroupCompressor, groupcompress.PyrexGroupCompressor):
            raise tests.TestNotApplicable('pure-python compressor does not handle compressor_settings')
        locations, old_block = self.make_block(self._texts)
        manager = groupcompress._LazyGroupContentManager(old_block, get_compressor_settings=lambda: dict(max_bytes_to_index=32))
        gc = manager._make_group_compressor()
        self.assertEqual(32, gc._delta_index._max_bytes_to_index)
        self.add_key_to_manager((b'key3',), locations, old_block, manager)
        self.add_key_to_manager((b'key4',), locations, old_block, manager)
        action, last_byte, total_bytes = manager._check_rebuild_action()
        self.assertEqual('rebuild', action)
        manager._rebuild_block()
        new_block = manager._block
        self.assertIsNot(old_block, new_block)
        self.assertTrue(old_block._content_length < new_block._content_length)

    def test_check_is_well_utilized_all_keys(self):
        block, manager = self.make_block_and_full_manager(self._texts)
        self.assertFalse(manager.check_is_well_utilized())
        manager._full_enough_block_size = block._content_length
        self.assertTrue(manager.check_is_well_utilized())
        manager._full_enough_block_size = block._content_length + 1
        self.assertFalse(manager.check_is_well_utilized())
        manager._full_enough_mixed_block_size = block._content_length
        self.assertFalse(manager.check_is_well_utilized())

    def test_check_is_well_utilized_mixed_keys(self):
        texts = {}
        f1k1 = (b'f1', b'k1')
        f1k2 = (b'f1', b'k2')
        f2k1 = (b'f2', b'k1')
        f2k2 = (b'f2', b'k2')
        texts[f1k1] = self._texts[b'key1',]
        texts[f1k2] = self._texts[b'key2',]
        texts[f2k1] = self._texts[b'key3',]
        texts[f2k2] = self._texts[b'key4',]
        block, manager = self.make_block_and_full_manager(texts)
        self.assertFalse(manager.check_is_well_utilized())
        manager._full_enough_block_size = block._content_length
        self.assertTrue(manager.check_is_well_utilized())
        manager._full_enough_block_size = block._content_length + 1
        self.assertFalse(manager.check_is_well_utilized())
        manager._full_enough_mixed_block_size = block._content_length
        self.assertTrue(manager.check_is_well_utilized())

    def test_check_is_well_utilized_partial_use(self):
        locations, block = self.make_block(self._texts)
        manager = groupcompress._LazyGroupContentManager(block)
        manager._full_enough_block_size = block._content_length
        self.add_key_to_manager((b'key1',), locations, block, manager)
        self.add_key_to_manager((b'key2',), locations, block, manager)
        self.assertFalse(manager.check_is_well_utilized())
        self.add_key_to_manager((b'key4',), locations, block, manager)
        self.assertTrue(manager.check_is_well_utilized())