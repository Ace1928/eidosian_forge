from ._base import *
import operator as op
class LazyDBCacheBase(abc.ABC):

    def __init__(self, save_path=None, cache_name='lazycache', *args, **kwargs):
        self._cachename = cache_name
        self.save_path = save_path or _lazydb_default_cache_path
        fio.mkdirs(self.save_path)
        self.lock = threading.RLock()
        self._args = args
        self._kwargs = kwargs

    @property
    def cache_file(self):
        return self._cachename + '.lazydb'

    @property
    def cache_filepath(self):
        return fio.join(self.save_path, self.cache_file)

    @property
    def exists(self):
        return fio.exists(self.cache_filepath)

    @abc.abstractmethod
    def dumps(self, data, *args, **kwargs):
        pass

    @abc.abstractmethod
    def loads(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def restore(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def save(self, db, *args, **kwargs):
        pass