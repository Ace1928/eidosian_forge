from __future__ import annotations
import functools
import itertools
import json
import logging
import math
import os
from collections import defaultdict
from operator import mul
from typing import TYPE_CHECKING
from monty.design_patterns import cached_class
from pymatgen.core import Species, get_el_sp
from pymatgen.util.due import Doi, due
@due.dcite(Doi('10.1021/ic102031h'), description='Data Mined Ionic Substitutions for the Discovery of New Compounds')
@cached_class
class SubstitutionProbability:
    """
    This class finds substitution probabilities given lists of atoms
    to substitute. The inputs make more sense if you look through the
    from_defaults static method.

    The substitution prediction algorithm is presented in:
    Hautier, G., Fischer, C., Ehrlacher, V., Jain, A., and Ceder, G. (2011)
    Data Mined Ionic Substitutions for the Discovery of New Compounds.
    Inorganic Chemistry, 50(2), 656-663. doi:10.1021/ic102031h
    """

    def __init__(self, lambda_table=None, alpha=-5):
        """
        Args:
            lambda_table:
                json table of the weight functions lambda if None,
                will use the default lambda.json table
            alpha:
                weight function for never observed substitutions.
        """
        if lambda_table is not None:
            self._lambda_table = lambda_table
        else:
            module_dir = os.path.dirname(__file__)
            json_file = f'{module_dir}/data/lambda.json'
            with open(json_file) as file:
                self._lambda_table = json.load(file)
        self.alpha = alpha
        self._l = {}
        self.species = set()
        for row in self._lambda_table:
            if 'D1+' not in row:
                s1 = Species.from_str(row[0])
                s2 = Species.from_str(row[1])
                self.species.add(s1)
                self.species.add(s2)
                self._l[frozenset([s1, s2])] = float(row[2])
        self.Z = 0
        self._px = defaultdict(float)
        for s1, s2 in itertools.product(self.species, repeat=2):
            value = math.exp(self.get_lambda(s1, s2))
            self._px[s1] += value / 2
            self._px[s2] += value / 2
            self.Z += value

    def get_lambda(self, s1, s2):
        """
        Args:
            s1 (Element/Species/str/int): Describes Ion in 1st Structure
            s2 (Element/Species/str/int): Describes Ion in 2nd Structure.

        Returns:
            Lambda values
        """
        k = frozenset([get_el_sp(s1), get_el_sp(s2)])
        return self._l.get(k, self.alpha)

    def get_px(self, sp):
        """
        Args:
            sp (Species/Element): Species.

        Returns:
            Probability
        """
        return self._px[get_el_sp(sp)]

    def prob(self, s1, s2):
        """
        Gets the probability of 2 species substitution. Not used by the
        structure predictor.

        Returns:
            Probability of s1 and s2 substitution.
        """
        return math.exp(self.get_lambda(s1, s2)) / self.Z

    def cond_prob(self, s1, s2):
        """
        Conditional probability of substituting s1 for s2.

        Args:
            s1:
                The *variable* specie
            s2:
                The *fixed* specie

        Returns:
            Conditional probability used by structure predictor.
        """
        return math.exp(self.get_lambda(s1, s2)) / self.get_px(s2)

    def pair_corr(self, s1, s2):
        """
        Pair correlation of two species.

        Returns:
            The pair correlation of 2 species
        """
        return math.exp(self.get_lambda(s1, s2)) * self.Z / (self.get_px(s1) * self.get_px(s2))

    def cond_prob_list(self, l1, l2):
        """
        Find the probabilities of 2 lists. These should include ALL species.
        This is the probability conditional on l2.

        Args:
            l1, l2:
                lists of species

        Returns:
            The conditional probability (assuming these species are in
            l2)
        """
        assert len(l1) == len(l2)
        p = 1
        for s1, s2 in zip(l1, l2):
            p *= self.cond_prob(s1, s2)
        return p

    def as_dict(self):
        """Returns: MSONable dict."""
        return {'name': type(self).__name__, 'version': __version__, 'init_args': {'lambda_table': self._l, 'alpha': self.alpha}, '@module': type(self).__module__, '@class': type(self).__name__}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): Dict representation.

        Returns:
            Class
        """
        return cls(**dct['init_args'])