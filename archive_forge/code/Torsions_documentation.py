from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.AtomPairs import Utils
from rdkit.Chem.rdMolDescriptors import (


  >>> from rdkit import Chem
  >>> m = Chem.MolFromSmiles('C=CC')
  >>> score=pyScorePath(m, (0, 1, 2), 3)
  >>> ExplainPathScore(score, 3)
  (('C', 1, 0), ('C', 2, 1), ('C', 1, 1))

  Again, it's order independent:

  >>> score = pyScorePath(m, (2, 1, 0), 3)
  >>> ExplainPathScore(score, 3)
  (('C', 1, 0), ('C', 2, 1), ('C', 1, 1))

  >>> m = Chem.MolFromSmiles('C=CO')
  >>> score=pyScorePath(m, (0, 1, 2), 3)
  >>> ExplainPathScore(score, 3)
  (('C', 1, 1), ('C', 2, 1), ('O', 1, 0))

  >>> m = Chem.MolFromSmiles('OC=CO')
  >>> score=pyScorePath(m, (0, 1, 2, 3), 4)
  >>> ExplainPathScore(score, 4)
  (('O', 1, 0), ('C', 2, 1), ('C', 2, 1), ('O', 1, 0))

  >>> m = Chem.MolFromSmiles('CC=CO')
  >>> score=pyScorePath(m, (0, 1, 2, 3), 4)
  >>> ExplainPathScore(score, 4)
  (('C', 1, 0), ('C', 2, 1), ('C', 2, 1), ('O', 1, 0))


  >>> m = Chem.MolFromSmiles('C=CC(=O)O')
  >>> score=pyScorePath(m, (0, 1, 2, 3), 4)
  >>> ExplainPathScore(score, 4)
  (('C', 1, 1), ('C', 2, 1), ('C', 3, 1), ('O', 1, 1))
  >>> score=pyScorePath(m, (0, 1, 2, 4), 4)
  >>> ExplainPathScore(score, 4)
  (('C', 1, 1), ('C', 2, 1), ('C', 3, 1), ('O', 1, 0))


  >>> m = Chem.MolFromSmiles('OOOO')
  >>> score=pyScorePath(m, (0, 1, 2), 3)
  >>> ExplainPathScore(score, 3)
  (('O', 1, 0), ('O', 2, 0), ('O', 2, 0))
  >>> score=pyScorePath(m, (0, 1, 2, 3), 4)
  >>> ExplainPathScore(score, 4)
  (('O', 1, 0), ('O', 2, 0), ('O', 2, 0), ('O', 1, 0))

  