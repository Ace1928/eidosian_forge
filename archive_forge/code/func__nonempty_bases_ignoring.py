@staticmethod
def _nonempty_bases_ignoring(base_tree, ignoring):
    return list(filter(None, [[b for b in bases if b is not ignoring] for bases in base_tree]))