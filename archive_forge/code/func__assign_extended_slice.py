from functools import total_ordering
def _assign_extended_slice(self, start, stop, step, valueList):
    """Assign an extended slice by re-assigning individual items"""
    indexList = range(start, stop, step)
    if len(valueList) != len(indexList):
        raise ValueError('attempt to assign sequence of size %d to extended slice of size %d' % (len(valueList), len(indexList)))
    for i, val in zip(indexList, valueList):
        self._set_single(i, val)