from .lib import TestBase, FileCreator
from smmap.mman import (
from smmap.buf import SlidingWindowMapBuffer
from random import randint
from time import time
import sys
import os
class TestBuf(TestBase):

    def test_basics(self):
        with FileCreator(self.k_window_test_size, 'buffer_test') as fc:
            c = man_optimal.make_cursor(fc.path)
            self.assertRaises(ValueError, SlidingWindowMapBuffer, type(c)())
            self.assertRaises(ValueError, SlidingWindowMapBuffer, c, fc.size)
            buf = SlidingWindowMapBuffer()
            assert buf.cursor() is None
            buf.end_access()
            buf.end_access()
            assert len(buf) == 0
            offset = 100
            assert buf.begin_access(c, fc.size) == False
            assert buf.begin_access(c, offset) == True
            assert len(buf) == fc.size - offset
            assert buf.cursor().is_valid()
            assert buf.begin_access() == True
            assert len(buf) == fc.size
            assert buf.cursor().is_valid()
            with open(fc.path, 'rb') as fp:
                data = fp.read()
            assert data[offset] == buf[0]
            assert data[offset:offset * 2] == buf[0:offset]
            assert buf[-1] == buf[len(buf) - 1]
            assert buf[-10:] == buf[len(buf) - 10:len(buf)]
            buf.end_access()
            assert not buf.cursor().is_valid()
            assert buf.cursor().is_associated()
            assert buf.begin_access() == True and buf.cursor().is_valid()
            del buf
            del c
            assert man_optimal.num_file_handles() == 1
            max_num_accesses = 100
            fd = os.open(fc.path, os.O_RDONLY)
            for item in (fc.path, fd):
                for manager, man_id in ((man_optimal, 'optimal'), (man_worst_case, 'worst case'), (static_man, 'static optimal')):
                    buf = SlidingWindowMapBuffer(manager.make_cursor(item))
                    assert manager.num_file_handles() == 1
                    for access_mode in range(2):
                        num_accesses_left = max_num_accesses
                        num_bytes = 0
                        fsize = fc.size
                        st = time()
                        buf.begin_access()
                        while num_accesses_left:
                            num_accesses_left -= 1
                            if access_mode:
                                ofs_start = randint(0, fsize)
                                ofs_end = randint(ofs_start, fsize)
                                d = buf[ofs_start:ofs_end]
                                assert len(d) == ofs_end - ofs_start
                                assert d == data[ofs_start:ofs_end]
                                num_bytes += len(d)
                                del d
                            else:
                                pos = randint(0, fsize)
                                assert buf[pos] == data[pos]
                                num_bytes += 1
                        buf.end_access()
                        assert manager.num_file_handles()
                        assert manager.collect()
                        assert manager.num_file_handles() == 0
                        elapsed = max(time() - st, 0.001)
                        mb = float(1000 * 1000)
                        mode_str = access_mode and 'slice' or 'single byte'
                        print('%s: Made %i random %s accesses to buffer created from %s reading a total of %f mb in %f s (%f mb/s)' % (man_id, max_num_accesses, mode_str, type(item), num_bytes / mb, elapsed, num_bytes / mb / elapsed), file=sys.stderr)
                    del buf
            os.close(fd)