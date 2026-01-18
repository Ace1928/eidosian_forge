from sympy.core import Basic, Dict, sympify, Tuple
from sympy.core.numbers import Integer
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import _sympify
from sympy.functions.combinatorial.numbers import bell
from sympy.matrices import zeros
from sympy.sets.sets import FiniteSet, Union
from sympy.utilities.iterables import flatten, group
from sympy.utilities.misc import as_int
from collections import defaultdict
class IntegerPartition(Basic):
    """
    This class represents an integer partition.

    Explanation
    ===========

    In number theory and combinatorics, a partition of a positive integer,
    ``n``, also called an integer partition, is a way of writing ``n`` as a
    list of positive integers that sum to n. Two partitions that differ only
    in the order of summands are considered to be the same partition; if order
    matters then the partitions are referred to as compositions. For example,
    4 has five partitions: [4], [3, 1], [2, 2], [2, 1, 1], and [1, 1, 1, 1];
    the compositions [1, 2, 1] and [1, 1, 2] are the same as partition
    [2, 1, 1].

    See Also
    ========

    sympy.utilities.iterables.partitions,
    sympy.utilities.iterables.multiset_partitions

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Partition_%28number_theory%29
    """
    _dict = None
    _keys = None

    def __new__(cls, partition, integer=None):
        """
        Generates a new IntegerPartition object from a list or dictionary.

        Explanation
        ===========

        The partition can be given as a list of positive integers or a
        dictionary of (integer, multiplicity) items. If the partition is
        preceded by an integer an error will be raised if the partition
        does not sum to that given integer.

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> a = IntegerPartition([5, 4, 3, 1, 1])
        >>> a
        IntegerPartition(14, (5, 4, 3, 1, 1))
        >>> print(a)
        [5, 4, 3, 1, 1]
        >>> IntegerPartition({1:3, 2:1})
        IntegerPartition(5, (2, 1, 1, 1))

        If the value that the partition should sum to is given first, a check
        will be made to see n error will be raised if there is a discrepancy:

        >>> IntegerPartition(10, [5, 4, 3, 1])
        Traceback (most recent call last):
        ...
        ValueError: The partition is not valid

        """
        if integer is not None:
            integer, partition = (partition, integer)
        if isinstance(partition, (dict, Dict)):
            _ = []
            for k, v in sorted(partition.items(), reverse=True):
                if not v:
                    continue
                k, v = (as_int(k), as_int(v))
                _.extend([k] * v)
            partition = tuple(_)
        else:
            partition = tuple(sorted(map(as_int, partition), reverse=True))
        sum_ok = False
        if integer is None:
            integer = sum(partition)
            sum_ok = True
        else:
            integer = as_int(integer)
        if not sum_ok and sum(partition) != integer:
            raise ValueError('Partition did not add to %s' % integer)
        if any((i < 1 for i in partition)):
            raise ValueError('All integer summands must be greater than one')
        obj = Basic.__new__(cls, Integer(integer), Tuple(*partition))
        obj.partition = list(partition)
        obj.integer = integer
        return obj

    def prev_lex(self):
        """Return the previous partition of the integer, n, in lexical order,
        wrapping around to [1, ..., 1] if the partition is [n].

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> p = IntegerPartition([4])
        >>> print(p.prev_lex())
        [3, 1]
        >>> p.partition > p.prev_lex().partition
        True
        """
        d = defaultdict(int)
        d.update(self.as_dict())
        keys = self._keys
        if keys == [1]:
            return IntegerPartition({self.integer: 1})
        if keys[-1] != 1:
            d[keys[-1]] -= 1
            if keys[-1] == 2:
                d[1] = 2
            else:
                d[keys[-1] - 1] = d[1] = 1
        else:
            d[keys[-2]] -= 1
            left = d[1] + keys[-2]
            new = keys[-2]
            d[1] = 0
            while left:
                new -= 1
                if left - new >= 0:
                    d[new] += left // new
                    left -= d[new] * new
        return IntegerPartition(self.integer, d)

    def next_lex(self):
        """Return the next partition of the integer, n, in lexical order,
        wrapping around to [n] if the partition is [1, ..., 1].

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> p = IntegerPartition([3, 1])
        >>> print(p.next_lex())
        [4]
        >>> p.partition < p.next_lex().partition
        True
        """
        d = defaultdict(int)
        d.update(self.as_dict())
        key = self._keys
        a = key[-1]
        if a == self.integer:
            d.clear()
            d[1] = self.integer
        elif a == 1:
            if d[a] > 1:
                d[a + 1] += 1
                d[a] -= 2
            else:
                b = key[-2]
                d[b + 1] += 1
                d[1] = (d[b] - 1) * b
                d[b] = 0
        elif d[a] > 1:
            if len(key) == 1:
                d.clear()
                d[a + 1] = 1
                d[1] = self.integer - a - 1
            else:
                a1 = a + 1
                d[a1] += 1
                d[1] = d[a] * a - a1
                d[a] = 0
        else:
            b = key[-2]
            b1 = b + 1
            d[b1] += 1
            need = d[b] * b + d[a] * a - b1
            d[a] = d[b] = 0
            d[1] = need
        return IntegerPartition(self.integer, d)

    def as_dict(self):
        """Return the partition as a dictionary whose keys are the
        partition integers and the values are the multiplicity of that
        integer.

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> IntegerPartition([1]*3 + [2] + [3]*4).as_dict()
        {1: 3, 2: 1, 3: 4}
        """
        if self._dict is None:
            groups = group(self.partition, multiple=False)
            self._keys = [g[0] for g in groups]
            self._dict = dict(groups)
        return self._dict

    @property
    def conjugate(self):
        """
        Computes the conjugate partition of itself.

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> a = IntegerPartition([6, 3, 3, 2, 1])
        >>> a.conjugate
        [5, 4, 3, 1, 1, 1]
        """
        j = 1
        temp_arr = list(self.partition) + [0]
        k = temp_arr[0]
        b = [0] * k
        while k > 0:
            while k > temp_arr[j]:
                b[k - 1] = j
                k -= 1
            j += 1
        return b

    def __lt__(self, other):
        """Return True if self is less than other when the partition
        is listed from smallest to biggest.

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> a = IntegerPartition([3, 1])
        >>> a < a
        False
        >>> b = a.next_lex()
        >>> a < b
        True
        >>> a == b
        False
        """
        return list(reversed(self.partition)) < list(reversed(other.partition))

    def __le__(self, other):
        """Return True if self is less than other when the partition
        is listed from smallest to biggest.

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> a = IntegerPartition([4])
        >>> a <= a
        True
        """
        return list(reversed(self.partition)) <= list(reversed(other.partition))

    def as_ferrers(self, char='#'):
        """
        Prints the ferrer diagram of a partition.

        Examples
        ========

        >>> from sympy.combinatorics.partitions import IntegerPartition
        >>> print(IntegerPartition([1, 1, 5]).as_ferrers())
        #####
        #
        #
        """
        return '\n'.join([char * i for i in self.partition])

    def __str__(self):
        return str(list(self.partition))