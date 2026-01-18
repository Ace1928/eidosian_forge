import operator
from typing import Any, Optional
class MinSegmentTree(SegmentTree):

    def __init__(self, capacity: int):
        super(MinSegmentTree, self).__init__(capacity=capacity, operation=min)

    def min(self, start: int=0, end: Optional[Any]=None) -> Any:
        """Returns min(arr[start], ...,  arr[end])"""
        return self.reduce(start, end)