import unittest
from numba.tests.support import TestCase
@jitclass
class ListLoopIterator:
    counter: Counter
    items: List[float]

    def __init__(self, items: List[float]):
        self.items = items
        self.counter = Counter()

    def get(self) -> float:
        idx = self.counter.get() % len(self.items)
        return self.items[idx]