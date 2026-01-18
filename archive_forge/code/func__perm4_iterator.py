from . import matrix
def _perm4_iterator():
    for v0 in range(4):
        for v1 in range(4):
            if v1 != v0:
                for v2 in range(4):
                    if v2 != v0 and v2 != v1:
                        yield (v0, v1, v2, 6 - v0 - v1 - v2)