from operator import inv
@staticmethod
def S4():
    """"
        All permutations in S4

        >>> len(list(Perm4.S4()))
        24
        """
    for p in S4_tuples:
        yield Perm4(p)