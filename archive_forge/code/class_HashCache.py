import os
import stat
import time
from .. import atomicfile, errors
from .. import filters as _mod_filters
from .. import osutils, trace
class HashCache:
    """Cache for looking up file SHA-1.

    Files are considered to match the cached value if the fingerprint
    of the file has not changed.  This includes its mtime, ctime,
    device number, inode number, and size.  This should catch
    modifications or replacement of the file by a new one.

    This may not catch modifications that do not change the file's
    size and that occur within the resolution window of the
    timestamps.  To handle this we specifically do not cache files
    which have changed since the start of the present second, since
    they could undetectably change again.

    This scheme may fail if the machine's clock steps backwards.
    Don't do that.

    This does not canonicalize the paths passed in; that should be
    done by the caller.

    _cache
        Indexed by path, points to a two-tuple of the SHA-1 of the file.
        and its fingerprint.

    stat_count
        number of times files have been statted

    hit_count
        number of times files have been retrieved from the cache, avoiding a
        re-read

    miss_count
        number of misses (times files have been completely re-read)
    """
    needs_write = False

    def __init__(self, root, cache_file_name, mode=None, content_filter_stack_provider=None):
        """Create a hash cache in base dir, and set the file mode to mode.

        :param content_filter_stack_provider: a function that takes a
            path (relative to the top of the tree) and a file-id as
            parameters and returns a stack of ContentFilters.
            If None, no content filtering is performed.
        """
        if not isinstance(root, str):
            raise ValueError('Base dir for hashcache must be text')
        self.root = root
        self.hit_count = 0
        self.miss_count = 0
        self.stat_count = 0
        self.danger_count = 0
        self.removed_count = 0
        self.update_count = 0
        self._cache = {}
        self._mode = mode
        self._cache_file_name = cache_file_name
        self._filter_provider = content_filter_stack_provider

    def cache_file_name(self):
        return self._cache_file_name

    def clear(self):
        """Discard all cached information.

        This does not reset the counters."""
        if self._cache:
            self.needs_write = True
            self._cache = {}

    def scan(self):
        """Scan all files and remove entries where the cache entry is obsolete.

        Obsolete entries are those where the file has been modified or deleted
        since the entry was inserted.
        """

        def inode_order(path_and_cache):
            return path_and_cache[1][1][3]
        for path, cache_val in sorted(self._cache.items(), key=inode_order):
            abspath = osutils.pathjoin(self.root, path)
            fp = self._fingerprint(abspath)
            self.stat_count += 1
            if not fp or cache_val[1] != fp:
                self.removed_count += 1
                self.needs_write = True
                del self._cache[path]

    def get_sha1(self, path, stat_value=None):
        """Return the sha1 of a file.
        """
        abspath = osutils.pathjoin(self.root, path)
        self.stat_count += 1
        file_fp = self._fingerprint(abspath, stat_value)
        if not file_fp:
            if path in self._cache:
                self.removed_count += 1
                self.needs_write = True
                del self._cache[path]
            return None
        if path in self._cache:
            cache_sha1, cache_fp = self._cache[path]
        else:
            cache_sha1, cache_fp = (None, None)
        if cache_fp == file_fp:
            self.hit_count += 1
            return cache_sha1
        self.miss_count += 1
        mode = file_fp[FP_MODE_COLUMN]
        if stat.S_ISREG(mode):
            if self._filter_provider is None:
                filters = []
            else:
                filters = self._filter_provider(path=path)
            digest = self._really_sha1_file(abspath, filters)
        elif stat.S_ISLNK(mode):
            target = osutils.readlink(abspath)
            digest = osutils.sha_string(target.encode('UTF-8'))
        else:
            raise errors.BzrError('file %r: unknown file stat mode: %o' % (abspath, mode))
        cutoff = self._cutoff_time()
        if file_fp[FP_MTIME_COLUMN] >= cutoff or file_fp[FP_CTIME_COLUMN] >= cutoff:
            self.danger_count += 1
            if cache_fp:
                self.removed_count += 1
                self.needs_write = True
                del self._cache[path]
        else:
            self.update_count += 1
            self.needs_write = True
            self._cache[path] = (digest, file_fp)
        return digest

    def _really_sha1_file(self, abspath, filters):
        """Calculate the SHA1 of a file by reading the full text"""
        return _mod_filters.internal_size_sha_file_byname(abspath, filters)[1]

    def write(self):
        """Write contents of cache to file."""
        with atomicfile.AtomicFile(self.cache_file_name(), 'wb', new_mode=self._mode) as outf:
            outf.write(CACHE_HEADER)
            for path, c in self._cache.items():
                line_info = [path.encode('utf-8'), b'// ', c[0], b' ']
                line_info.append(b'%d %d %d %d %d %d' % c[1])
                line_info.append(b'\n')
                outf.write(b''.join(line_info))
            self.needs_write = False

    def read(self):
        """Reinstate cache from file.

        Overwrites existing cache.

        If the cache file has the wrong version marker, this just clears
        the cache."""
        self._cache = {}
        fn = self.cache_file_name()
        try:
            inf = open(fn, 'rb', buffering=65000)
        except OSError as e:
            trace.mutter('failed to open %s: %s', fn, str(e))
            self.needs_write = True
            return
        with inf:
            hdr = inf.readline()
            if hdr != CACHE_HEADER:
                trace.mutter('cache header marker not found at top of %s; discarding cache', fn)
                self.needs_write = True
                return
            for l in inf:
                pos = l.index(b'// ')
                path = l[:pos].decode('utf-8')
                if path in self._cache:
                    trace.warning('duplicated path %r in cache' % path)
                    continue
                pos += 3
                fields = l[pos:].split(b' ')
                if len(fields) != 7:
                    trace.warning('bad line in hashcache: %r' % l)
                    continue
                sha1 = fields[0]
                if len(sha1) != 40:
                    trace.warning('bad sha1 in hashcache: %r' % sha1)
                    continue
                fp = tuple(map(int, fields[1:]))
                self._cache[path] = (sha1, fp)
        self.needs_write = False

    def _cutoff_time(self):
        """Return cutoff time.

        Files modified more recently than this time are at risk of being
        undetectably modified and so can't be cached.
        """
        return int(time.time()) - 3

    def _fingerprint(self, abspath, stat_value=None):
        if stat_value is None:
            try:
                stat_value = os.lstat(abspath)
            except OSError:
                return None
        if stat.S_ISDIR(stat_value.st_mode):
            return None
        return (stat_value.st_size, int(stat_value.st_mtime), int(stat_value.st_ctime), stat_value.st_ino, stat_value.st_dev, stat_value.st_mode)