import itertools
import copy
import numbers
from Bio.Phylo import BaseTree
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Align import substitution_matrices
class ParsimonyTreeConstructor(TreeConstructor):
    """Parsimony tree constructor.

    :Parameters:
        searcher : TreeSearcher
            tree searcher to search the best parsimony tree.
        starting_tree : Tree
            starting tree provided to the searcher.

    Examples
    --------
    We will load an alignment, and then load various trees which have already been computed from it::

      >>> from Bio import AlignIO, Phylo
      >>> aln = AlignIO.read(open('TreeConstruction/msa.phy'), 'phylip')
      >>> print(aln)
      Alignment with 5 rows and 13 columns
      AACGTGGCCACAT Alpha
      AAGGTCGCCACAC Beta
      CAGTTCGCCACAA Gamma
      GAGATTTCCGCCT Delta
      GAGATCTCCGCCC Epsilon

    Load a starting tree::

      >>> starting_tree = Phylo.read('TreeConstruction/nj.tre', 'newick')
      >>> print(starting_tree)
      Tree(rooted=False, weight=1.0)
          Clade(branch_length=0.0, name='Inner3')
              Clade(branch_length=0.01421, name='Inner2')
                  Clade(branch_length=0.23927, name='Inner1')
                      Clade(branch_length=0.08531, name='Epsilon')
                      Clade(branch_length=0.13691, name='Delta')
                  Clade(branch_length=0.2923, name='Alpha')
              Clade(branch_length=0.07477, name='Beta')
              Clade(branch_length=0.17523, name='Gamma')

    Build the Parsimony tree from the starting tree::

      >>> scorer = Phylo.TreeConstruction.ParsimonyScorer()
      >>> searcher = Phylo.TreeConstruction.NNITreeSearcher(scorer)
      >>> constructor = Phylo.TreeConstruction.ParsimonyTreeConstructor(searcher, starting_tree)
      >>> pars_tree = constructor.build_tree(aln)
      >>> print(pars_tree)
      Tree(rooted=True, weight=1.0)
          Clade(branch_length=0.0)
              Clade(branch_length=0.19732999999999998, name='Inner1')
                  Clade(branch_length=0.13691, name='Delta')
                  Clade(branch_length=0.08531, name='Epsilon')
              Clade(branch_length=0.04194000000000003, name='Inner2')
                  Clade(branch_length=0.01421, name='Inner3')
                      Clade(branch_length=0.17523, name='Gamma')
                      Clade(branch_length=0.07477, name='Beta')
                  Clade(branch_length=0.2923, name='Alpha')

    Same example, using the new Alignment class::

      >>> from Bio import Align, Phylo
      >>> alignment = Align.read(open('TreeConstruction/msa.phy'), 'phylip')
      >>> print(alignment)
      Alpha             0 AACGTGGCCACAT 13
      Beta              0 AAGGTCGCCACAC 13
      Gamma             0 CAGTTCGCCACAA 13
      Delta             0 GAGATTTCCGCCT 13
      Epsilon           0 GAGATCTCCGCCC 13
      <BLANKLINE>

    Load a starting tree::

      >>> starting_tree = Phylo.read('TreeConstruction/nj.tre', 'newick')
      >>> print(starting_tree)
      Tree(rooted=False, weight=1.0)
          Clade(branch_length=0.0, name='Inner3')
              Clade(branch_length=0.01421, name='Inner2')
                  Clade(branch_length=0.23927, name='Inner1')
                      Clade(branch_length=0.08531, name='Epsilon')
                      Clade(branch_length=0.13691, name='Delta')
                  Clade(branch_length=0.2923, name='Alpha')
              Clade(branch_length=0.07477, name='Beta')
              Clade(branch_length=0.17523, name='Gamma')

    Build the Parsimony tree from the starting tree::

      >>> scorer = Phylo.TreeConstruction.ParsimonyScorer()
      >>> searcher = Phylo.TreeConstruction.NNITreeSearcher(scorer)
      >>> constructor = Phylo.TreeConstruction.ParsimonyTreeConstructor(searcher, starting_tree)
      >>> pars_tree = constructor.build_tree(alignment)
      >>> print(pars_tree)
      Tree(rooted=True, weight=1.0)
          Clade(branch_length=0.0)
              Clade(branch_length=0.19732999999999998, name='Inner1')
                  Clade(branch_length=0.13691, name='Delta')
                  Clade(branch_length=0.08531, name='Epsilon')
              Clade(branch_length=0.04194000000000003, name='Inner2')
                  Clade(branch_length=0.01421, name='Inner3')
                      Clade(branch_length=0.17523, name='Gamma')
                      Clade(branch_length=0.07477, name='Beta')
                  Clade(branch_length=0.2923, name='Alpha')

    """

    def __init__(self, searcher, starting_tree=None):
        """Initialize the class."""
        self.searcher = searcher
        self.starting_tree = starting_tree

    def build_tree(self, alignment):
        """Build the tree.

        :Parameters:
            alignment : MultipleSeqAlignment
                multiple sequence alignment to calculate parsimony tree.

        """
        if self.starting_tree is None:
            dtc = DistanceTreeConstructor(DistanceCalculator('identity'), 'upgma')
            self.starting_tree = dtc.build_tree(alignment)
        return self.searcher.search(self.starting_tree, alignment)