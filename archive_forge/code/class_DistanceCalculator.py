import itertools
import copy
import numbers
from Bio.Phylo import BaseTree
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Align import substitution_matrices
class DistanceCalculator:
    """Calculates the distance matrix from a DNA or protein sequence alignment.

    This class calculates the distance matrix from a multiple sequence alignment
    of DNA or protein sequences, and the given name of the substitution model.

    Currently only scoring matrices are used.

    :Parameters:
        model : str
            Name of the model matrix to be used to calculate distance.
            The attribute ``dna_models`` contains the available model
            names for DNA sequences and ``protein_models`` for protein
            sequences.

    Examples
    --------
    Loading a small PHYLIP alignment from which to compute distances::

      >>> from Bio.Phylo.TreeConstruction import DistanceCalculator
      >>> from Bio import AlignIO
      >>> aln = AlignIO.read(open('TreeConstruction/msa.phy'), 'phylip')
      >>> print(aln)  # doctest:+NORMALIZE_WHITESPACE
      Alignment with 5 rows and 13 columns
      AACGTGGCCACAT Alpha
      AAGGTCGCCACAC Beta
      CAGTTCGCCACAA Gamma
      GAGATTTCCGCCT Delta
      GAGATCTCCGCCC Epsilon

    DNA calculator with 'identity' model::

      >>> calculator = DistanceCalculator('identity')
      >>> dm = calculator.get_distance(aln)
      >>> print(dm)  # doctest:+NORMALIZE_WHITESPACE
        Alpha   0.000000
        Beta    0.230769    0.000000
        Gamma   0.384615    0.230769    0.000000
        Delta   0.538462    0.538462    0.538462    0.000000
        Epsilon 0.615385    0.384615    0.461538    0.153846    0.000000
            Alpha   Beta    Gamma   Delta   Epsilon

    Protein calculator with 'blosum62' model::

      >>> calculator = DistanceCalculator('blosum62')
      >>> dm = calculator.get_distance(aln)
      >>> print(dm)  # doctest:+NORMALIZE_WHITESPACE
      Alpha   0.000000
      Beta    0.369048    0.000000
      Gamma   0.493976    0.250000    0.000000
      Delta   0.585366    0.547619    0.566265    0.000000
      Epsilon 0.700000    0.355556    0.488889    0.222222    0.000000
          Alpha   Beta    Gamma   Delta   Epsilon

    Same calculation, using the new Alignment object::

      >>> from Bio.Phylo.TreeConstruction import DistanceCalculator
      >>> from Bio import Align
      >>> aln = Align.read('TreeConstruction/msa.phy', 'phylip')
      >>> print(aln)  # doctest:+NORMALIZE_WHITESPACE
      Alpha             0 AACGTGGCCACAT 13
      Beta              0 AAGGTCGCCACAC 13
      Gamma             0 CAGTTCGCCACAA 13
      Delta             0 GAGATTTCCGCCT 13
      Epsilon           0 GAGATCTCCGCCC 13
      <BLANKLINE>

    DNA calculator with 'identity' model::

      >>> calculator = DistanceCalculator('identity')
      >>> dm = calculator.get_distance(aln)
      >>> print(dm)  # doctest:+NORMALIZE_WHITESPACE
      Alpha   0.000000
      Beta    0.230769    0.000000
      Gamma   0.384615    0.230769    0.000000
      Delta   0.538462    0.538462    0.538462    0.000000
      Epsilon 0.615385    0.384615    0.461538    0.153846    0.000000
          Alpha   Beta    Gamma   Delta   Epsilon

    Protein calculator with 'blosum62' model::

      >>> calculator = DistanceCalculator('blosum62')
      >>> dm = calculator.get_distance(aln)
      >>> print(dm)  # doctest:+NORMALIZE_WHITESPACE
      Alpha   0.000000
      Beta    0.369048    0.000000
      Gamma   0.493976    0.250000    0.000000
      Delta   0.585366    0.547619    0.566265    0.000000
      Epsilon 0.700000    0.355556    0.488889    0.222222    0.000000
          Alpha   Beta    Gamma   Delta   Epsilon

    """
    protein_alphabet = set('ABCDEFGHIKLMNPQRSTVWXYZ')
    dna_models = []
    protein_models = []
    names = substitution_matrices.load()
    for name in names:
        matrix = substitution_matrices.load(name)
        if name == 'NUC.4.4':
            name = 'blastn'
        else:
            name = name.lower()
        if protein_alphabet.issubset(set(matrix.alphabet)):
            protein_models.append(name)
        else:
            dna_models.append(name)
    del protein_alphabet
    del name
    del names
    del matrix
    models = ['identity'] + dna_models + protein_models

    def __init__(self, model='identity', skip_letters=None):
        """Initialize with a distance model."""
        if skip_letters:
            self.skip_letters = skip_letters
        elif model == 'identity':
            self.skip_letters = ()
        else:
            self.skip_letters = ('-', '*')
        if model == 'identity':
            self.scoring_matrix = None
        elif model in self.models:
            if model == 'blastn':
                name = 'NUC.4.4'
            else:
                name = model.upper()
            self.scoring_matrix = substitution_matrices.load(name)
        else:
            raise ValueError('Model not supported. Available models: ' + ', '.join(self.models))

    def _pairwise(self, seq1, seq2):
        """Calculate pairwise distance from two sequences (PRIVATE).

        Returns a value between 0 (identical sequences) and 1 (completely
        different, or seq1 is an empty string.)
        """
        score = 0
        max_score = 0
        if self.scoring_matrix is None:
            score = sum((l1 == l2 for l1, l2 in zip(seq1, seq2) if l1 not in self.skip_letters and l2 not in self.skip_letters))
            max_score = len(seq1)
        else:
            max_score1 = 0
            max_score2 = 0
            for i in range(0, len(seq1)):
                l1 = seq1[i]
                l2 = seq2[i]
                if l1 in self.skip_letters or l2 in self.skip_letters:
                    continue
                try:
                    max_score1 += self.scoring_matrix[l1, l1]
                except IndexError:
                    raise ValueError(f"Bad letter '{l1}' in sequence '{seq1.id}' at position '{i}'") from None
                try:
                    max_score2 += self.scoring_matrix[l2, l2]
                except IndexError:
                    raise ValueError(f"Bad letter '{l2}' in sequence '{seq2.id}' at position '{i}'") from None
                score += self.scoring_matrix[l1, l2]
            max_score = max(max_score1, max_score2)
        if max_score == 0:
            return 1
        return 1 - score / max_score

    def get_distance(self, msa):
        """Return a DistanceMatrix for an Alignment or MultipleSeqAlignment object.

        :Parameters:
            msa : Alignment or MultipleSeqAlignment object representing a
                DNA or protein multiple sequence alignment.

        """
        if isinstance(msa, Alignment):
            names = [s.id for s in msa.sequences]
            dm = DistanceMatrix(names)
            n = len(names)
            for i1 in range(n):
                for i2 in range(i1):
                    dm[names[i1], names[i2]] = self._pairwise(msa[i1], msa[i2])
        elif isinstance(msa, MultipleSeqAlignment):
            names = [s.id for s in msa]
            dm = DistanceMatrix(names)
            for seq1, seq2 in itertools.combinations(msa, 2):
                dm[seq1.id, seq2.id] = self._pairwise(seq1, seq2)
        else:
            raise TypeError('Must provide an Alignment object or a MultipleSeqAlignment object.')
        return dm