from Bio.Data import IUPACData
from typing import Dict, List, Optional
class CodonTable:
    """A codon-table, or genetic code."""
    forward_table: Dict[str, str] = {}
    back_table: Dict[str, str] = {}
    start_codons: List[str] = []
    stop_codons: List[str] = []

    def __init__(self, nucleotide_alphabet: Optional[str]=None, protein_alphabet: Optional[str]=None, forward_table: Dict[str, str]=forward_table, back_table: Dict[str, str]=back_table, start_codons: List[str]=start_codons, stop_codons: List[str]=stop_codons) -> None:
        """Initialize the class."""
        self.nucleotide_alphabet = nucleotide_alphabet
        self.protein_alphabet = protein_alphabet
        self.forward_table = forward_table
        self.back_table = back_table
        self.start_codons = start_codons
        self.stop_codons = stop_codons

    def __str__(self):
        """Return a simple text representation of the codon table.

        e.g.::

            >>> import Bio.Data.CodonTable
            >>> print(Bio.Data.CodonTable.standard_dna_table)
            Table 1 Standard, SGC0
            <BLANKLINE>
              |  T      |  C      |  A      |  G      |
            --+---------+---------+---------+---------+--
            T | TTT F   | TCT S   | TAT Y   | TGT C   | T
            T | TTC F   | TCC S   | TAC Y   | TGC C   | C
            ...
            G | GTA V   | GCA A   | GAA E   | GGA G   | A
            G | GTG V   | GCG A   | GAG E   | GGG G   | G
            --+---------+---------+---------+---------+--
            >>> print(Bio.Data.CodonTable.generic_by_id[1])
            Table 1 Standard, SGC0
            <BLANKLINE>
              |  U      |  C      |  A      |  G      |
            --+---------+---------+---------+---------+--
            U | UUU F   | UCU S   | UAU Y   | UGU C   | U
            U | UUC F   | UCC S   | UAC Y   | UGC C   | C
            ...
            G | GUA V   | GCA A   | GAA E   | GGA G   | A
            G | GUG V   | GCG A   | GAG E   | GGG G   | G
            --+---------+---------+---------+---------+--
        """
        if self.id:
            answer = 'Table %i' % self.id
        else:
            answer = 'Table ID unknown'
        if self.names:
            answer += ' ' + ', '.join([x for x in self.names if x])
        letters = self.nucleotide_alphabet
        if letters is not None and 'T' in letters:
            letters = 'TCAG'
        else:
            letters = 'UCAG'
        answer += '\n\n'
        answer += '  |' + '|'.join((f'  {c2}      ' for c2 in letters)) + '|'
        answer += '\n--+' + '+'.join(('---------' for c2 in letters)) + '+--'
        for c1 in letters:
            for c3 in letters:
                line = c1 + ' |'
                for c2 in letters:
                    codon = c1 + c2 + c3
                    line += f' {codon}'
                    if codon in self.stop_codons:
                        line += ' Stop|'
                    else:
                        try:
                            amino = self.forward_table[codon]
                        except KeyError:
                            amino = '?'
                        except TranslationError:
                            amino = '?'
                        if codon in self.start_codons:
                            line += f' {amino}(s)|'
                        else:
                            line += f' {amino}   |'
                line += ' ' + c3
                answer += '\n' + line
            answer += '\n--+' + '+'.join(('---------' for c2 in letters)) + '+--'
        return answer