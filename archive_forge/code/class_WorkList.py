from heapq import heappop, heappush
from .lru_cache import LRUCache
class WorkList:

    def __init__(self):
        self.pq = []

    def add(self, item):
        dt, cmt = item
        heappush(self.pq, (-dt, cmt))

    def get(self):
        item = heappop(self.pq)
        if item:
            pr, cmt = item
            return (-pr, cmt)
        return None

    def iter(self):
        for pr, cmt in self.pq:
            yield (-pr, cmt)