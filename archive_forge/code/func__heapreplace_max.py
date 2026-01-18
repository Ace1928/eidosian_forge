def _heapreplace_max(heap, item):
    """Maxheap version of a heappop followed by a heappush."""
    returnitem = heap[0]
    heap[0] = item
    _siftup_max(heap, 0)
    return returnitem