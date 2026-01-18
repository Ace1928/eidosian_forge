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