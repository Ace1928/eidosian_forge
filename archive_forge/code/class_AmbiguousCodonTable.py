from Bio.Data import IUPACData
from typing import Dict, List, Optional
class AmbiguousCodonTable(CodonTable):
    """Base codon table for ambiguous sequences."""

    def __init__(self, codon_table, ambiguous_nucleotide_alphabet, ambiguous_nucleotide_values, ambiguous_protein_alphabet, ambiguous_protein_values):
        """Initialize the class."""
        CodonTable.__init__(self, ambiguous_nucleotide_alphabet, ambiguous_protein_alphabet, AmbiguousForwardTable(codon_table.forward_table, ambiguous_nucleotide_values, ambiguous_protein_values), codon_table.back_table, list_ambiguous_codons(codon_table.start_codons, ambiguous_nucleotide_values), list_ambiguous_codons(codon_table.stop_codons, ambiguous_nucleotide_values))
        self._codon_table = codon_table

    def __getattr__(self, name):
        """Forward attribute lookups to the original table."""
        return getattr(self._codon_table, name)