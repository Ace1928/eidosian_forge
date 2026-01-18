from __future__ import annotations
class HashedSeq(list):
    """Hashed Sequence.

    Type used for hash() to make sure the hash is not generated
    multiple times.
    """
    __slots__ = 'hashvalue'

    def __init__(self, *seq):
        self[:] = seq
        self.hashvalue = hash(seq)

    def __hash__(self):
        return self.hashvalue