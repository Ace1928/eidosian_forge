from sage.all import QQ, vector, matrix, VectorSpace
class Vector3(Vector):

    def __init__(self, entries):
        assert len(entries) == 3
        Vector.__init__(self, entries)