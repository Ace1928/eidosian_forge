from operator import inv
class Perm4Basic:
    """
    Class Perm4Basic: A permutation of {0,1,2,3}.

    A permutation can be initialized with a length 4 dictionary or a
    4-tuple or a length 2 dictionary; in the latter case the sign of
    the permutation can be specified by setting "sign=0" (even) or
    "sign=1" (odd).  The default sign is odd, since odd permutations
    describe orientation-preserving gluings.

    This is the original version of Perm4, which was tablized for
    speed reasons.
    """

    def __init__(self, init, sign=1):
        self.dict = {}
        if isinstance(init, Perm4Basic) or len(init) == 4:
            for i in range(4):
                self.dict[i] = init[i]
        else:
            self.dict = init
            v = list(init.items())
            x = opposite[v[0][0], v[1][0]]
            y = opposite[v[0][1], v[1][1]]
            self.dict[x[0]] = y[sign]
            self.dict[x[1]] = y[1 - sign]

    def image(self, bitmap):
        """
        A subset of {0,1,2,3} can be represented by a bitmap.  This
        computes the bitmap of the image subset.

        >>> Perm4Basic([2, 3, 1, 0]).image(10)
        9
        """
        image = 0
        for i in range(4):
            if bitmap & 1 << i:
                image = image | 1 << self.dict[i]
        return image

    def __repr__(self):
        return str(self.tuple())

    def __call__(self, a_tuple):
        """
        P((i, ... ,j)) returns the image tuple (P(i), ... , P(j))
        """
        image = []
        for i in a_tuple:
            image.append(self.dict[i])
        return tuple(image)

    def __getitem__(self, index):
        """
        P[i] returns the image of i, P(i)

        >>> Perm4Basic([2, 3, 1, 0])[3]
        0
        """
        return self.dict[index]

    def __mul__(self, other):
        """
        P*Q is the composition P*Q(i) = P(Q(i))

        >>> P = Perm4Basic((2, 3, 1, 0))
        >>> Q = Perm4Basic((1, 0, 2, 3))
        >>> P * Q
        (3, 2, 1, 0)
        >>> Q * P
        (2, 3, 0, 1)
        """
        composition = {}
        for i in range(4):
            composition[i] = self.dict[other.dict[i]]
        return Perm4Basic(composition)

    def __invert__(self):
        """
        inv(P) is the inverse permutation

        >>> inv(Perm4Basic([2, 1, 3, 0]))
        (3, 1, 0, 2)
        """
        inverse = {}
        for i in range(4):
            inverse[self.dict[i]] = i
        return Perm4Basic(inverse)

    def sign(self):
        """
        sign(P) is the sign: 0 for even, 1 for odd

        >>> Perm4Basic([0, 1, 3, 2]).sign()
        1
        >>> Perm4Basic([1, 0, 3, 2]).sign()
        0
        """
        sign = 0
        for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
            sign = sign ^ (self.dict[i] < self.dict[j])
        return sign

    def tuple(self):
        """
        P.tuple() returns a tuple representing the permutation P

        >>> Perm4Basic([1, 2, 0, 3]).tuple()
        (1, 2, 0, 3)
        """
        return tuple((self.dict[i] for i in range(4)))