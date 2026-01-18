import itertools
import math
import operator
import random
from functools import reduce
class RGroups:
    """Holds a collection of sidechains for the given RGroup"""

    def __init__(self, sidechains):
        """Sidechains -> RGroups
         sidechains: the list of Sidechains that make up the potential
                     sidechains at this rgroup position"""
        self.sidechains = sidechains
        self.rejected = []
        self.initial_size = len(sidechains)

    def count(self):
        """Returns the number of possible sidechains"""
        return len(self.sidechains)

    def randomize(self):
        """Randomly shuffles the sidechains and reset the goodness counts"""
        random.shuffle(self.sidechains)
        for s in self.sidechains:
            s.good_count = 0

    def effectiveness(self):
        """-> return the current effectiveness of this collection
        effectiveness is the number of items left divided by the 
        initial amount"""
        return len(self.sidechains) / float(self.initial_size)

    def chunk_size(self, num_chunks):
        """num_chunks -> return the number of sidechains in each chunk
        if the sidechains are split into num_chunks chunks"""
        return int(math.ceil(float(len(self.sidechains)) / num_chunks))

    def chunk(self, chunk_idx, num_chunks):
        """chunk_idx, num)chunks -> RGroups
        return the chunk_idxth chunk given num_chunks total chunks"""
        assert chunk_idx >= 0 and chunk_idx < num_chunks, '%s %s' % (chunk_idx, num_chunks)
        n = self.chunk_size(num_chunks)
        return RGroups(self.sidechains[chunk_idx * n:(chunk_idx + 1) * n])

    def prune(self, fractionToKeep):
        """fractionToKeep -> Sort the sidechains from the most often 
        found if good products to the least, and keep the best 
        fractionToKeep percentage"""
        assert 0 < fractionToKeep <= 1.0, 'fractionToKeep: %s' % fractionToKeep
        self.sidechains.sort(key=lambda x: x.good_count, reverse=True)
        fragment_index = int(len(self.sidechains) * fractionToKeep + 0.5)
        self.rejected += self.sidechains[fragment_index:]
        self.sidechains = self.sidechains[:fragment_index]