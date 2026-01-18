from Bio.Data import IUPACData
from typing import Dict, List, Optional
class NCBICodonTable(CodonTable):
    """Codon table for generic nucleotide sequences."""
    nucleotide_alphabet: Optional[str] = None
    protein_alphabet = IUPACData.protein_letters

    def __init__(self, id, names, table, start_codons, stop_codons):
        """Initialize the class."""
        self.id = id
        self.names = names
        self.forward_table = table
        self.back_table = make_back_table(table, stop_codons[0])
        self.start_codons = start_codons
        self.stop_codons = stop_codons

    def __repr__(self):
        """Represent the NCBI codon table class as a string for debugging."""
        return f'{self.__class__.__name__}(id={self.id!r}, names={self.names!r}, ...)'