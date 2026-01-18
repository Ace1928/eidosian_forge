import gevent
from gevent import pool
from dulwich.objects import (
from dulwich.object_store import (
class GreenThreadsObjectStoreIterator(ObjectStoreIterator):
    """ObjectIterator that works on top of an ObjectStore.

    Same implementation as object_store.ObjectStoreIterator
    except we use gevent to parallelize object retrieval.
    """

    def __init__(self, store, shas, finder, concurrency=1):
        self.finder = finder
        self.p = pool.Pool(size=concurrency)
        super(GreenThreadsObjectStoreIterator, self).__init__(store, shas)

    def retrieve(self, args):
        sha, path = args
        return (self.store[sha], path)

    def __iter__(self):
        for sha, path in self.p.imap_unordered(self.retrieve, self.itershas()):
            yield (sha, path)

    def __len__(self):
        if len(self._shas) > 0:
            return len(self._shas)
        while len(self.finder.objects_to_send):
            jobs = []
            for _ in range(0, len(self.finder.objects_to_send)):
                jobs.append(self.p.spawn(self.finder.next))
            gevent.joinall(jobs)
            for j in jobs:
                if j.value is not None:
                    self._shas.append(j.value)
        return len(self._shas)