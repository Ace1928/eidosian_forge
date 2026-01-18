from .cartan_type import CartanType
from sympy.core.basic import Atom
def add_as_roots(self, root1, root2):
    """Add two roots together if and only if their sum is also a root

        It takes as input two vectors which should be roots.  It then computes
        their sum and checks if it is in the list of all possible roots.  If it
        is, it returns the sum.  Otherwise it returns a string saying that the
        sum is not a root.

        Examples
        ========

        >>> from sympy.liealgebras.root_system import RootSystem
        >>> c = RootSystem("A3")
        >>> c.add_as_roots([1, 0, -1, 0], [0, 0, 1, -1])
        [1, 0, 0, -1]
        >>> c.add_as_roots([1, -1, 0, 0], [0, 0, -1, 1])
        'The sum of these two roots is not a root'

        """
    alpha = self.all_roots()
    newroot = [r1 + r2 for r1, r2 in zip(root1, root2)]
    if newroot in alpha.values():
        return newroot
    else:
        return 'The sum of these two roots is not a root'