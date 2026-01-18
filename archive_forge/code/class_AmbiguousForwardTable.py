from Bio.Data import IUPACData
from typing import Dict, List, Optional
class AmbiguousForwardTable:
    """Forward table for translation of ambiguous nucleotide sequences."""

    def __init__(self, forward_table, ambiguous_nucleotide, ambiguous_protein):
        """Initialize the class."""
        self.forward_table = forward_table
        self.ambiguous_nucleotide = ambiguous_nucleotide
        self.ambiguous_protein = ambiguous_protein
        inverted = {}
        for name, val in ambiguous_protein.items():
            for c in val:
                x = inverted.get(c, {})
                x[name] = 1
                inverted[c] = x
        for name, val in inverted.items():
            inverted[name] = list(val)
        self._inverted = inverted
        self._cache = {}

    def __contains__(self, codon):
        """Check if codon works as key for ambiguous forward_table.

        Only returns 'True' if forward_table[codon] returns a value.
        """
        try:
            self.__getitem__(codon)
            return True
        except (KeyError, TranslationError):
            return False

    def get(self, codon, failobj=None):
        """Implement get for dictionary-like behaviour."""
        try:
            return self.__getitem__(codon)
        except KeyError:
            return failobj

    def __getitem__(self, codon):
        """Implement dictionary-like behaviour for AmbiguousForwardTable.

        forward_table[codon] will either return an amino acid letter,
        or throws a KeyError (if codon does not encode an amino acid)
        or a TranslationError (if codon does encode for an amino acid,
        but either is also a stop codon or does encode several amino acids,
        for which no unique letter is available in the given alphabet.
        """
        try:
            x = self._cache[codon]
        except KeyError:
            pass
        else:
            if x is TranslationError:
                raise TranslationError(codon)
            if x is KeyError:
                raise KeyError(codon)
            return x
        try:
            x = self.forward_table[codon]
            self._cache[codon] = x
            return x
        except KeyError:
            pass
        try:
            possible = list_possible_proteins(codon, self.forward_table, self.ambiguous_nucleotide)
        except KeyError:
            self._cache[codon] = KeyError
            raise KeyError(codon) from None
        except TranslationError:
            self._cache[codon] = TranslationError
            raise TranslationError(codon)
        assert len(possible) > 0, 'unambiguous codons must code'
        if len(possible) == 1:
            self._cache[codon] = possible[0]
            return possible[0]
        ambiguous_possible = {}
        for amino in possible:
            for term in self._inverted[amino]:
                ambiguous_possible[term] = ambiguous_possible.get(term, 0) + 1
        n = len(possible)
        possible = []
        for amino, val in ambiguous_possible.items():
            if val == n:
                possible.append(amino)
        if len(possible) == 0:
            self._cache[codon] = TranslationError
            raise TranslationError(codon)
        possible.sort(key=lambda x: (len(self.ambiguous_protein[x]), x))
        x = possible[0]
        self._cache[codon] = x
        return x