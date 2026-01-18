import heapq
def _siftdown(self, start_pos, pos):
    """Restore invariant. keep swapping with parent until smaller.

        Built to mimic code for heapq._siftdown
        only updating position dict too.
        """
    heap, position = (self.heap, self.position)
    newitem = heap[pos]
    while pos > start_pos:
        parent_pos = pos - 1 >> 1
        parent = heap[parent_pos]
        if not newitem < parent:
            break
        heap[pos] = parent
        position[parent] = pos
        pos = parent_pos
    heap[pos] = newitem
    position[newitem] = pos