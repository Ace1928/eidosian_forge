import rpy2.rlike.indexing as rli
from typing import Any
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
def __delslice__(self, i, j):
    super(TaggedList, self).__delslice__(i, j)
    self.__tags.__delslice__(i, j)