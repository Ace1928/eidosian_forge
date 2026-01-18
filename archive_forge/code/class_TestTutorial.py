from .lib import TestBase
class TestTutorial(TestBase):

    def test_example(self):
        import smmap
        mman = smmap.SlidingWindowMapManager()
        assert mman.num_file_handles() == 0
        assert mman.mapped_memory_size() == 0
        import smmap.test.lib
        with smmap.test.lib.FileCreator(1024 * 1024 * 8, 'test_file') as fc:
            c = mman.make_cursor(fc.path)
            assert c.is_associated()
            assert not c.is_valid()
            assert c.use_region().is_valid()
            assert c.size()
            c.buffer()[0]
            c.buffer()[1:10]
            c.buffer()[c.size() - 1]
            assert c.ofs_begin() < c.ofs_end()
            assert c.includes_ofs(100)
            assert not c.use_region(fc.size, 100).is_valid()
            assert c.use_region(100).is_valid()
            c.unuse_region()
            assert not c.is_valid()
            buf = smmap.SlidingWindowMapBuffer(mman.make_cursor(fc.path))
            assert buf.cursor().is_valid()
            buf[0]
            buf[-1]
            buf[-10:]
            buf.end_access()
            assert not buf.cursor().is_valid()
            assert buf.begin_access(offset=10)