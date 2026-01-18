from operator import inv
@staticmethod
def KleinFour():
    """
        Z/2 x Z/2 as a subgroup of A4.

        >>> len(list(Perm4.KleinFour()))
        4
        """
    for p in KleinFour_tuples:
        yield Perm4(p)