import bisect
from collections import defaultdict
import mmap
import os
import sys
import tempfile
import threading
from .context import reduction, assert_spawning
from . import util
class Heap(object):
    _alignment = 8
    _DISCARD_FREE_SPACE_LARGER_THAN = 4 * 1024 ** 2
    _DOUBLE_ARENA_SIZE_UNTIL = 4 * 1024 ** 2

    def __init__(self, size=mmap.PAGESIZE):
        self._lastpid = os.getpid()
        self._lock = threading.Lock()
        self._size = size
        self._lengths = []
        self._len_to_seq = {}
        self._start_to_block = {}
        self._stop_to_block = {}
        self._allocated_blocks = defaultdict(set)
        self._arenas = []
        self._pending_free_blocks = []
        self._n_mallocs = 0
        self._n_frees = 0

    @staticmethod
    def _roundup(n, alignment):
        mask = alignment - 1
        return n + mask & ~mask

    def _new_arena(self, size):
        length = self._roundup(max(self._size, size), mmap.PAGESIZE)
        if self._size < self._DOUBLE_ARENA_SIZE_UNTIL:
            self._size *= 2
        util.info('allocating a new mmap of length %d', length)
        arena = Arena(length)
        self._arenas.append(arena)
        return (arena, 0, length)

    def _discard_arena(self, arena):
        length = arena.size
        if length < self._DISCARD_FREE_SPACE_LARGER_THAN:
            return
        blocks = self._allocated_blocks.pop(arena)
        assert not blocks
        del self._start_to_block[arena, 0]
        del self._stop_to_block[arena, length]
        self._arenas.remove(arena)
        seq = self._len_to_seq[length]
        seq.remove((arena, 0, length))
        if not seq:
            del self._len_to_seq[length]
            self._lengths.remove(length)

    def _malloc(self, size):
        i = bisect.bisect_left(self._lengths, size)
        if i == len(self._lengths):
            return self._new_arena(size)
        else:
            length = self._lengths[i]
            seq = self._len_to_seq[length]
            block = seq.pop()
            if not seq:
                del self._len_to_seq[length], self._lengths[i]
        arena, start, stop = block
        del self._start_to_block[arena, start]
        del self._stop_to_block[arena, stop]
        return block

    def _add_free_block(self, block):
        arena, start, stop = block
        try:
            prev_block = self._stop_to_block[arena, start]
        except KeyError:
            pass
        else:
            start, _ = self._absorb(prev_block)
        try:
            next_block = self._start_to_block[arena, stop]
        except KeyError:
            pass
        else:
            _, stop = self._absorb(next_block)
        block = (arena, start, stop)
        length = stop - start
        try:
            self._len_to_seq[length].append(block)
        except KeyError:
            self._len_to_seq[length] = [block]
            bisect.insort(self._lengths, length)
        self._start_to_block[arena, start] = block
        self._stop_to_block[arena, stop] = block

    def _absorb(self, block):
        arena, start, stop = block
        del self._start_to_block[arena, start]
        del self._stop_to_block[arena, stop]
        length = stop - start
        seq = self._len_to_seq[length]
        seq.remove(block)
        if not seq:
            del self._len_to_seq[length]
            self._lengths.remove(length)
        return (start, stop)

    def _remove_allocated_block(self, block):
        arena, start, stop = block
        blocks = self._allocated_blocks[arena]
        blocks.remove((start, stop))
        if not blocks:
            self._discard_arena(arena)

    def _free_pending_blocks(self):
        while True:
            try:
                block = self._pending_free_blocks.pop()
            except IndexError:
                break
            self._add_free_block(block)
            self._remove_allocated_block(block)

    def free(self, block):
        if os.getpid() != self._lastpid:
            raise ValueError('My pid ({0:n}) is not last pid {1:n}'.format(os.getpid(), self._lastpid))
        if not self._lock.acquire(False):
            self._pending_free_blocks.append(block)
        else:
            try:
                self._n_frees += 1
                self._free_pending_blocks()
                self._add_free_block(block)
                self._remove_allocated_block(block)
            finally:
                self._lock.release()

    def malloc(self, size):
        if size < 0:
            raise ValueError('Size {0:n} out of range'.format(size))
        if sys.maxsize <= size:
            raise OverflowError('Size {0:n} too large'.format(size))
        if os.getpid() != self._lastpid:
            self.__init__()
        with self._lock:
            self._n_mallocs += 1
            self._free_pending_blocks()
            size = self._roundup(max(size, 1), self._alignment)
            arena, start, stop = self._malloc(size)
            real_stop = start + size
            if real_stop < stop:
                self._add_free_block((arena, real_stop, stop))
            self._allocated_blocks[arena].add((start, real_stop))
            return (arena, start, real_stop)