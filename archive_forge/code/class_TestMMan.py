from .lib import TestBase, FileCreator
from smmap.mman import (
from smmap.util import align_to_mmap
from random import randint
from time import time
import os
import sys
from copy import copy
class TestMMan(TestBase):

    def test_cursor(self):
        with FileCreator(self.k_window_test_size, 'cursor_test') as fc:
            man = SlidingWindowMapManager()
            ci = WindowCursor(man)
            assert not ci.is_valid()
            assert not ci.is_associated()
            assert ci.size() == 0
            cv = man.make_cursor(fc.path)
            assert not cv.is_valid()
            assert cv.is_associated()
            assert cv.file_size() == fc.size
            assert cv.path() == fc.path
        cio = copy(cv)
        assert not cio.is_valid() and cio.is_associated()
        assert not ci.is_associated()
        ci.assign(cv)
        assert not ci.is_valid() and ci.is_associated()
        cv.unuse_region()
        cv.unuse_region()
        cv._destroy()
        WindowCursor(man)._destroy()

    def test_memory_manager(self):
        slide_man = SlidingWindowMapManager()
        static_man = StaticWindowMapManager()
        for man in (static_man, slide_man):
            assert man.num_file_handles() == 0
            assert man.num_open_files() == 0
            winsize_cmp_val = 0
            if isinstance(man, StaticWindowMapManager):
                winsize_cmp_val = -1
            assert man.window_size() > winsize_cmp_val
            assert man.mapped_memory_size() == 0
            assert man.max_mapped_memory_size() > 0
            man._collect_lru_region(0)
            man._collect_lru_region(10)
            assert man._collect_lru_region(sys.maxsize) == 0
            with FileCreator(self.k_window_test_size, 'manager_test') as fc:
                fd = os.open(fc.path, os.O_RDONLY)
                try:
                    for item in (fc.path, fd):
                        c = man.make_cursor(item)
                        assert c.path_or_fd() is item
                        assert c.use_region(10, 10).is_valid()
                        assert c.ofs_begin() == 10
                        assert c.size() == 10
                        with open(fc.path, 'rb') as fp:
                            assert c.buffer()[:] == fp.read(20)[10:]
                    if isinstance(item, int):
                        self.assertRaises(ValueError, c.path)
                    else:
                        self.assertRaises(ValueError, c.fd)
                finally:
                    os.close(fd)

    def test_memman_operation(self):
        with FileCreator(self.k_window_test_size, 'manager_operation_test') as fc:
            with open(fc.path, 'rb') as fp:
                data = fp.read()
            fd = os.open(fc.path, os.O_RDONLY)
            try:
                max_num_handles = 15
                for mtype, args in ((StaticWindowMapManager, (0, fc.size // 3, max_num_handles)), (SlidingWindowMapManager, (fc.size // 100, fc.size // 3, max_num_handles))):
                    for item in (fc.path, fd):
                        assert len(data) == fc.size
                        man = mtype(window_size=args[0], max_memory_size=args[1], max_open_handles=args[2])
                        c = man.make_cursor(item)
                        assert man.num_open_files() == 0
                        assert man.mapped_memory_size() == 0
                        base_offset = 5000
                        size = man.window_size() // 2
                        assert c.use_region(base_offset, size).is_valid()
                        rr = c.region()
                        assert rr.client_count() == 2
                        assert man.num_open_files() == 1
                        assert man.num_file_handles() == 1
                        assert man.mapped_memory_size() == rr.size()
                        assert c.ofs_begin() == base_offset
                        assert rr.ofs_begin() == 0
                        if man.window_size():
                            assert rr.size() == align_to_mmap(man.window_size(), True)
                        else:
                            assert rr.size() == fc.size
                        assert c.buffer()[:] == data[base_offset:base_offset + (size or c.size())]
                        nsize = (size or fc.size) - 10
                        assert c.use_region(0, nsize).is_valid()
                        assert c.region() == rr
                        assert man.num_file_handles() == 1
                        assert c.size() == nsize
                        assert c.ofs_begin() == 0
                        assert c.buffer()[:] == data[:nsize]
                        overshoot = 4000
                        base_offset = fc.size - (size or c.size()) + overshoot
                        assert c.use_region(base_offset, size).is_valid()
                        if man.window_size():
                            assert man.num_file_handles() == 2
                            assert c.size() < size
                            assert c.region() is not rr
                            assert rr.client_count() == 1
                        else:
                            assert c.size() < fc.size
                        rr = c.region()
                        assert rr.client_count() == 2
                        assert rr.ofs_begin() < c.ofs_begin()
                        assert rr.ofs_end() <= fc.size
                        assert c.buffer()[:] == data[base_offset:base_offset + (size or c.size())]
                        c.unuse_region()
                        assert not c.is_valid()
                        if man.window_size():
                            assert man.num_file_handles() == 2
                        max_random_accesses = 5000
                        num_random_accesses = max_random_accesses
                        memory_read = 0
                        st = time()
                        includes_ofs = c.includes_ofs
                        max_mapped_memory_size = man.max_mapped_memory_size()
                        max_file_handles = man.max_file_handles()
                        mapped_memory_size = man.mapped_memory_size
                        num_file_handles = man.num_file_handles
                        while num_random_accesses:
                            num_random_accesses -= 1
                            base_offset = randint(0, fc.size - 1)
                            if man.window_size():
                                assert max_mapped_memory_size >= mapped_memory_size()
                            assert max_file_handles >= num_file_handles()
                            assert c.use_region(base_offset, size or c.size()).is_valid()
                            csize = c.size()
                            assert c.buffer()[:] == data[base_offset:base_offset + csize]
                            memory_read += csize
                            assert includes_ofs(base_offset)
                            assert includes_ofs(base_offset + csize - 1)
                            assert not includes_ofs(base_offset + csize)
                        elapsed = max(time() - st, 0.001)
                        mb = float(1000 * 1000)
                        print('%s: Read %i mb of memory with %i random on cursor initialized with %s accesses in %fs (%f mb/s)\n' % (mtype, memory_read / mb, max_random_accesses, type(item), elapsed, memory_read / mb / elapsed), file=sys.stderr)
                        assert not c.use_region(fc.size, size).is_valid()
                        assert man.num_file_handles()
                        assert man.collect()
                        assert man.num_file_handles() == 0
            finally:
                os.close(fd)