class _IntervalTreeTester(IntervalTree):
    """
    A test rig for IntervalTree. It will keep a separate plain list of all
    entries added by :py:meth:`insert`. It offers methods to compare the result
    of a brute force search in the plain list against an interval tree search.
    It also offers methods to do many consistency checks on the red black tree.

    Run the test rig::

        sage: _IntervalTreeTester.run_test()

    """

    def __init__(self):
        self._entries = []
        super(_IntervalTreeTester, self).__init__()

    def insert(self, interval, value):
        self._entries.append((interval, value))
        super(_IntervalTreeTester, self).insert(interval, value)

    def brute_force_find(self, interval):
        """
        Search plain list as ground truth to compare against.
        """
        return [entry[1] for entry in self._entries if entry[0].overlaps(interval)]

    def check_find_result(self, interval):
        if set(self.find(interval)) != set(self.brute_force_find(interval)):
            raise Exception('Different results: %r %r' % (self.find(interval), self.brute_force_find(interval)))

    def check_consistency(self):
        from sage.all import Infinity
        if self._root.isRed:
            raise Exception('Red root')
        _IntervalTreeTester._recursively_check_consistency(self._root, -Infinity, +Infinity)

    @staticmethod
    def _recursively_check_consistency(node, l, r):
        from sage.all import Infinity
        if not node:
            return (-Infinity, 0)
        if not node.interval.lower() >= l:
            raise Exception('Node left  lower %r %r', node.interval.lower(), l)
        if not node.interval.lower() <= r:
            raise Exception('Node right lower %r %r', node.interval.lower(), r)
        left_max, left_depth = _IntervalTreeTester._recursively_check_consistency(node.children[LEFT], l, node.interval.lower())
        right_max, right_depth = _IntervalTreeTester._recursively_check_consistency(node.children[RIGHT], node.interval.lower(), r)
        if not max(left_max, right_max, node.interval.upper()) == node.max_value:
            raise Exception('Maximum incorrect')
        if left_depth != right_depth:
            raise Exception('Inconsistent black depths')
        if node.isRed:
            for child in node.children:
                if child and child.isRed:
                    raise Exception('Red node has red child')
        else:
            left_depth += 1
        return (node.max_value, left_depth)

    def print_tree(self):
        self.print_tree_recursively(self._root, 0)

    @staticmethod
    def print_tree_recursively(node, depth):
        if not node:
            return
        if not node.isRed:
            depth += 1
        _IntervalTreeTester.print_tree_recursively(node.children[0], depth)
        align = 6 * depth
        if node.isRed:
            align += 3
        print(align * ' ', end=' ')
        if node.isRed:
            print('R', end=' ')
        else:
            print('B', end=' ')
        print(node.interval.lower())
        _IntervalTreeTester.print_tree_recursively(node.children[1], depth)

    @staticmethod
    def run_test():
        from sage.all import RIF, sin, Infinity
        intervals = [RIF(sin(1.2 * i), sin(1.2 * i) + sin(1.43 * i) ** 2) for i in range(200)]
        t = _IntervalTreeTester()
        for i, interval in enumerate(intervals):
            t.insert(interval, i)
            if i % 50 == 0:
                for j in intervals:
                    t.check_find_result(j)
                t.check_consistency()
        num_true = len(intervals)
        num_have = len(t.find(RIF(-Infinity, Infinity)))
        if num_true != num_have:
            raise Exception('Inconsistent number of intervals: %d %d' % (num_true, num_have))