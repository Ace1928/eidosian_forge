from __future__ import annotations
import bisect
from copy import copy, deepcopy
from datetime import datetime
from math import log, pi, sqrt
from typing import TYPE_CHECKING, Any
from warnings import warn
import numpy as np
from monty.json import MSONable
from scipy import constants
from scipy.special import comb, erfc
from pymatgen.core.structure import Structure
from pymatgen.util.due import Doi, due
class EwaldMinimizer:
    """
    This class determines the manipulations that will minimize an Ewald matrix,
    given a list of possible manipulations. This class does not perform the
    manipulations on a structure, but will return the list of manipulations
    that should be done on one to produce the minimal structure. It returns the
    manipulations for the n lowest energy orderings. This class should be used
    to perform fractional species substitution or fractional species removal to
    produce a new structure. These manipulations create large numbers of
    candidate structures, and this class can be used to pick out those with the
    lowest Ewald sum.

    An alternative (possibly more intuitive) interface to this class is the
    order disordered structure transformation.

    Author - Will Richards
    """
    ALGO_FAST = 0
    ALGO_COMPLETE = 1
    ALGO_BEST_FIRST = 2
    ALGO_TIME_LIMIT = 3

    def __init__(self, matrix, m_list, num_to_return=1, algo=ALGO_FAST):
        """
        Args:
            matrix: A matrix of the Ewald sum interaction energies. This is stored
                in the class as a diagonally symmetric array and so
                self._matrix will not be the same as the input matrix.
            m_list: list of manipulations. each item is of the form
                (multiplication fraction, number_of_indices, indices, species)
                These are sorted such that the first manipulation contains the
                most permutations. this is actually evaluated last in the
                recursion since I'm using pop.
            num_to_return: The minimizer will find the number_returned lowest
                energy structures. This is likely to return a number of duplicate
                structures so it may be necessary to overestimate and then
                remove the duplicates later. (duplicate checking in this
                process is extremely expensive).
        """
        self._matrix = copy(matrix)
        for ii in range(len(self._matrix)):
            for jj in range(ii, len(self._matrix)):
                value = (self._matrix[ii, jj] + self._matrix[jj, ii]) / 2
                self._matrix[ii, jj] = value
                self._matrix[jj, ii] = value
        self._m_list = sorted(m_list, key=lambda x: comb(len(x[2]), x[1]), reverse=True)
        for mlist in self._m_list:
            if mlist[0] > 1:
                raise ValueError('multiplication fractions must be <= 1')
        self._current_minimum = float('inf')
        self._num_to_return = num_to_return
        self._algo = algo
        if algo == EwaldMinimizer.ALGO_COMPLETE:
            raise NotImplementedError('Complete algo not yet implemented for EwaldMinimizer')
        self._output_lists = []
        self._finished = False
        self._start_time = datetime.utcnow()
        self.minimize_matrix()
        self._best_m_list = self._output_lists[0][1]
        self._minimized_sum = self._output_lists[0][0]

    def minimize_matrix(self):
        """
        This method finds and returns the permutations that produce the lowest
        Ewald sum calls recursive function to iterate through permutations.
        """
        if self._algo in (EwaldMinimizer.ALGO_FAST, EwaldMinimizer.ALGO_BEST_FIRST):
            return self._recurse(self._matrix, self._m_list, set(range(len(self._matrix))))
        return None

    def add_m_list(self, matrix_sum, m_list):
        """
        This adds an m_list to the output_lists and updates the current
        minimum if the list is full.
        """
        if self._output_lists is None:
            self._output_lists = [[matrix_sum, m_list]]
        else:
            bisect.insort(self._output_lists, [matrix_sum, m_list])
        if self._algo == EwaldMinimizer.ALGO_BEST_FIRST and len(self._output_lists) == self._num_to_return:
            self._finished = True
        if len(self._output_lists) > self._num_to_return:
            self._output_lists.pop()
        if len(self._output_lists) == self._num_to_return:
            self._current_minimum = self._output_lists[-1][0]

    def best_case(self, matrix, m_list, indices_left):
        """
        Computes a best case given a matrix and manipulation list.

        Args:
            matrix: the current matrix (with some permutations already
                performed)
            m_list: [(multiplication fraction, number_of_indices, indices,
                species)] describing the manipulation
            indices: Set of indices which haven't had a permutation
                performed on them.
        """
        m_indices = []
        fraction_list = []
        for m in m_list:
            m_indices.extend(m[2])
            fraction_list.extend([m[0]] * m[1])
        indices = list(indices_left.intersection(m_indices))
        interaction_matrix = matrix[indices, :][:, indices]
        fractions = np.zeros(len(interaction_matrix)) + 1
        fractions[:len(fraction_list)] = fraction_list
        fractions = np.sort(fractions)
        sums = 2 * np.sum(matrix[indices], axis=1)
        sums = np.sort(sums)
        step1 = np.sort(interaction_matrix) * (1 - fractions)
        step2 = np.sort(np.sum(step1, axis=1))
        step3 = step2 * (1 - fractions)
        interaction_correction = np.sum(step3)
        if self._algo == self.ALGO_TIME_LIMIT:
            elapsed_time = datetime.utcnow() - self._start_time
            speedup_parameter = elapsed_time.total_seconds() / 1800
            avg_int = np.sum(interaction_matrix, axis=None)
            avg_frac = np.average(np.outer(1 - fractions, 1 - fractions))
            average_correction = avg_int * avg_frac
            interaction_correction = average_correction * speedup_parameter + interaction_correction * (1 - speedup_parameter)
        return np.sum(matrix) + np.inner(sums[::-1], fractions - 1) + interaction_correction

    @classmethod
    def get_next_index(cls, matrix, manipulation, indices_left):
        """
        Returns an index that should have the most negative effect on the
        matrix sum.
        """
        f = manipulation[0]
        indices = list(indices_left.intersection(manipulation[2]))
        sums = np.sum(matrix[indices], axis=1)
        return indices[sums.argmax(axis=0)] if f < 1 else indices[sums.argmin(axis=0)]

    def _recurse(self, matrix, m_list, indices, output_m_list=None):
        """
        This method recursively finds the minimal permutations using a binary
        tree search strategy.

        Args:
            matrix: The current matrix (with some permutations already
                performed).
            m_list: The list of permutations still to be performed
            indices: Set of indices which haven't had a permutation
                performed on them.
        """
        if self._finished:
            return
        if output_m_list is None:
            output_m_list = []
        while m_list[-1][1] == 0:
            m_list = copy(m_list)
            m_list.pop()
            if not m_list:
                matrix_sum = np.sum(matrix)
                if matrix_sum < self._current_minimum:
                    self.add_m_list(matrix_sum, output_m_list)
                return
        if m_list[-1][1] > len(indices.intersection(m_list[-1][2])):
            return
        if (len(m_list) == 1 or m_list[-1][1] > 1) and self.best_case(matrix, m_list, indices) > self._current_minimum:
            return
        index = self.get_next_index(matrix, m_list[-1], indices)
        m_list[-1][2].remove(index)
        matrix2 = np.copy(matrix)
        m_list2 = deepcopy(m_list)
        output_m_list2 = copy(output_m_list)
        matrix2[index, :] *= m_list[-1][0]
        matrix2[:, index] *= m_list[-1][0]
        output_m_list2.append([index, m_list[-1][3]])
        indices2 = copy(indices)
        indices2.remove(index)
        m_list2[-1][1] -= 1
        self._recurse(matrix2, m_list2, indices2, output_m_list2)
        self._recurse(matrix, m_list, indices, output_m_list)

    @property
    def best_m_list(self):
        """Returns: Best m_list found."""
        return self._best_m_list

    @property
    def minimized_sum(self):
        """Returns: Minimized sum."""
        return self._minimized_sum

    @property
    def output_lists(self):
        """Returns: output lists."""
        return self._output_lists