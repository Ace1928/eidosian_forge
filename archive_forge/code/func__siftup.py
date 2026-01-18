import heapq
def _siftup(self, pos):
    """Move smaller child up until hitting a leaf.

        Built to mimic code for heapq._siftup
        only updating position dict too.
        """
    heap, position = (self.heap, self.position)
    end_pos = len(heap)
    startpos = pos
    newitem = heap[pos]
    child_pos = (pos << 1) + 1
    while child_pos < end_pos:
        child = heap[child_pos]
        right_pos = child_pos + 1
        if right_pos < end_pos:
            right = heap[right_pos]
            if not child < right:
                child = right
                child_pos = right_pos
        heap[pos] = child
        position[child] = pos
        pos = child_pos
        child_pos = (pos << 1) + 1
    while pos > 0:
        parent_pos = pos - 1 >> 1
        parent = heap[parent_pos]
        if not newitem < parent:
            break
        heap[pos] = parent
        position[parent] = pos
        pos = parent_pos
    heap[pos] = newitem
    position[newitem] = pos