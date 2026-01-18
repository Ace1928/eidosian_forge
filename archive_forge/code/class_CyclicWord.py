from .graphs import ReducedGraph, Digraph, Poset
from collections import deque
import operator
class CyclicWord(Word):

    def cancel(self):
        Word.cancel(self)
        while len(self) > 0 and self[0] == -self[-1]:
            self.popleft()
            self.pop()

    def __mul__(self, other):
        raise ValueError('Cyclic words cannot be multiplied.')

    def __invert__(self):
        inverse = [-x for x in reversed(self)]
        return CyclicWord(inverse, alphabet=self.alphabet)

    def spun(self, start=0):
        """
        Generator for letters in cyclic order, starting at start.
        """
        N = len(self)
        for n in range(start, start + N):
            yield self[n % N]

    def invert(self):
        """
        Invert this cyclic word in place.
        """
        for n in range(len(self) // 2):
            self[n], self[-1 - n] = (self[-1 - n], self[n])
        map(operator.neg, self)

    def rewrite(self, ordering):
        seq = []
        for letter in self:
            if letter in ordering:
                seq.append(1 + ordering.index(letter))
            else:
                seq.append(-1 - ordering.index(-letter))
        return CyclicWord(seq)

    def shuffle(self, perm_dict={}):
        """
        Permute generators according to the supplied dictionary.
        Keys must be positive integers.  Values may be negative.
        The set of keys must equal the set of values up to sign.
        """
        abs_image = set(map(operator.abs, perm_dict.values()))
        if set(perm_dict) != abs_image:
            raise ValueError('Not a permutation!')
        for n in range(len(self)):
            x = self[n]
            self[n] = perm_dict.get(x, x) if x > 0 else -perm_dict.get(-x, -x)

    def powers(self, start=0):
        """
        Return a list of pairs (letter, power) for the exponential
        representation of the spun word, beginning at start.
        """
        result = []
        last_letter = self[start]
        count = 0
        for letter in self.spun(start):
            if letter == last_letter:
                count += 1
            else:
                result.append((last_letter, count))
                count = 1
                last_letter = letter
        result.append((last_letter, count))
        return result

    def complexity(self, size, ordering=[], spin=0):
        """
        Returns the complexity of the word relative to an extension of
        the ordering, and returns the extended ordering.  The
        lexicographical complexity is a list of integers, representing
        the ranking of each letter in the ordering.  The size is the
        total number of generators, which may be larger than the number
        of distinct generators that appear in the word.  If x appears
        in the ordering with rank r, then x^-1 has rank size + r.
        Unordered generators are added to the ordering as they are
        encountered in the word.
        """
        the_ordering = list(ordering)
        complexity = []
        for letter in self.spun(spin):
            if letter in the_ordering:
                complexity.append(the_ordering.index(letter))
            elif -letter in the_ordering:
                complexity.append(size + the_ordering.index(-letter))
            else:
                complexity.append(len(the_ordering))
                the_ordering.append(letter)
        return (Complexity(complexity), the_ordering)

    def minima(self, size, ordering=[]):
        """
        Return the minimal complexity of all rotations and inverted
        rotations, and a list of the words and orderings that realize
        the minimal complexity.
        """
        least = Complexity([])
        minima = []
        for word in (self, ~self):
            for n in range(len(self)):
                complexity, Xordering = word.complexity(size, ordering, spin=n)
                if complexity < least:
                    least = complexity
                    minima = [(CyclicWord(word.spun(n)), Xordering)]
                elif complexity == least:
                    minima.append((CyclicWord(word.spun(n)), Xordering))
        return (least, minima)