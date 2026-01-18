from Bio.Data import IUPACData
from typing import Dict, List, Optional
Implement dictionary-like behaviour for AmbiguousForwardTable.

        forward_table[codon] will either return an amino acid letter,
        or throws a KeyError (if codon does not encode an amino acid)
        or a TranslationError (if codon does encode for an amino acid,
        but either is also a stop codon or does encode several amino acids,
        for which no unique letter is available in the given alphabet.
        