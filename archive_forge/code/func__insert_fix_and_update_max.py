@staticmethod
def _insert_fix_and_update_max(node, leaf):
    violation, node = IntervalTree._insert_and_fix(node, leaf)
    node.update_max_value()
    return (violation, node)