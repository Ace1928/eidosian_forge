import heapq
class MappedQueue:
    """The MappedQueue class implements a min-heap with removal and update-priority.

    The min heap uses heapq as well as custom written _siftup and _siftdown
    methods to allow the heap positions to be tracked by an additional dict
    keyed by element to position. The smallest element can be popped in O(1) time,
    new elements can be pushed in O(log n) time, and any element can be removed
    or updated in O(log n) time. The queue cannot contain duplicate elements
    and an attempt to push an element already in the queue will have no effect.

    MappedQueue complements the heapq package from the python standard
    library. While MappedQueue is designed for maximum compatibility with
    heapq, it adds element removal, lookup, and priority update.

    Parameters
    ----------
    data : dict or iterable

    Examples
    --------

    A `MappedQueue` can be created empty, or optionally, given a dictionary
    of initial elements and priorities.  The methods `push`, `pop`,
    `remove`, and `update` operate on the queue.

    >>> colors_nm = {'red':665, 'blue': 470, 'green': 550}
    >>> q = MappedQueue(colors_nm)
    >>> q.remove('red')
    >>> q.update('green', 'violet', 400)
    >>> q.push('indigo', 425)
    True
    >>> [q.pop().element for i in range(len(q.heap))]
    ['violet', 'indigo', 'blue']

    A `MappedQueue` can also be initialized with a list or other iterable. The priority is assumed
    to be the sort order of the items in the list.

    >>> q = MappedQueue([916, 50, 4609, 493, 237])
    >>> q.remove(493)
    >>> q.update(237, 1117)
    >>> [q.pop() for i in range(len(q.heap))]
    [50, 916, 1117, 4609]

    An exception is raised if the elements are not comparable.

    >>> q = MappedQueue([100, 'a'])
    Traceback (most recent call last):
    ...
    TypeError: '<' not supported between instances of 'int' and 'str'

    To avoid the exception, use a dictionary to assign priorities to the elements.

    >>> q = MappedQueue({100: 0, 'a': 1 })

    References
    ----------
    .. [1] Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2001).
       Introduction to algorithms second edition.
    .. [2] Knuth, D. E. (1997). The art of computer programming (Vol. 3).
       Pearson Education.
    """

    def __init__(self, data=None):
        """Priority queue class with updatable priorities."""
        if data is None:
            self.heap = []
        elif isinstance(data, dict):
            self.heap = [_HeapElement(v, k) for k, v in data.items()]
        else:
            self.heap = list(data)
        self.position = {}
        self._heapify()

    def _heapify(self):
        """Restore heap invariant and recalculate map."""
        heapq.heapify(self.heap)
        self.position = {elt: pos for pos, elt in enumerate(self.heap)}
        if len(self.heap) != len(self.position):
            raise AssertionError('Heap contains duplicate elements')

    def __len__(self):
        return len(self.heap)

    def push(self, elt, priority=None):
        """Add an element to the queue."""
        if priority is not None:
            elt = _HeapElement(priority, elt)
        if elt in self.position:
            return False
        pos = len(self.heap)
        self.heap.append(elt)
        self.position[elt] = pos
        self._siftdown(0, pos)
        return True

    def pop(self):
        """Remove and return the smallest element in the queue."""
        elt = self.heap[0]
        del self.position[elt]
        if len(self.heap) == 1:
            self.heap.pop()
            return elt
        last = self.heap.pop()
        self.heap[0] = last
        self.position[last] = 0
        self._siftup(0)
        return elt

    def update(self, elt, new, priority=None):
        """Replace an element in the queue with a new one."""
        if priority is not None:
            new = _HeapElement(priority, new)
        pos = self.position[elt]
        self.heap[pos] = new
        del self.position[elt]
        self.position[new] = pos
        self._siftup(pos)

    def remove(self, elt):
        """Remove an element from the queue."""
        try:
            pos = self.position[elt]
            del self.position[elt]
        except KeyError:
            raise
        if pos == len(self.heap) - 1:
            self.heap.pop()
            return
        last = self.heap.pop()
        self.heap[pos] = last
        self.position[last] = pos
        self._siftup(pos)

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