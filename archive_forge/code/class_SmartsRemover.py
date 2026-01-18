from rdkit import Chem
from rdkit.VLib.Transform import TransformNode
class SmartsRemover(TransformNode):
    """ transforms molecules by removing atoms matching smarts patterns

  Assumptions:

    - inputs are molecules


  Sample Usage:
    >>> smis = ['C1CCC1.C=O','C1CCC1C=O','CCC=O.C=O','NCC=O.C=O.CN']
    >>> mols = [Chem.MolFromSmiles(x) for x in smis]
    >>> from rdkit.VLib.Supply import SupplyNode
    >>> suppl = SupplyNode(contents=mols)
    >>> ms = [x for x in suppl]
    >>> len(ms)
    4

    We can pass in SMARTS strings:
    >>> smas = ['C=O','CN']
    >>> tform = SmartsRemover(patterns=smas)
    >>> tform.AddParent(suppl)
    >>> ms = [x for x in tform]
    >>> len(ms)
    4
    >>> Chem.MolToSmiles(ms[0])
    'C1CCC1'
    >>> Chem.MolToSmiles(ms[1])
    'O=CC1CCC1'
    >>> Chem.MolToSmiles(ms[2])
    'CCC=O'
    >>> Chem.MolToSmiles(ms[3])
    'NCC=O'

    We can also remove pieces of the molecule that are not complete
    fragments:
    >>> tform.Destroy()
    >>> smas = ['C=O','CN']
    >>> smas = [Chem.MolFromSmarts(x) for x in smas]
    >>> tform = SmartsRemover(patterns=smas,wholeFragments=0)
    >>> tform.AddParent(suppl)
    >>> ms = [x for x in tform]
    >>> len(ms)
    4
    >>> Chem.MolToSmiles(ms[0])
    'C1CCC1'
    >>> Chem.MolToSmiles(ms[1])
    'C1CCC1'
    >>> Chem.MolToSmiles(ms[3])
    ''

    Or patterns themselves:
    >>> tform.Destroy()
    >>> smas = ['C=O','CN']
    >>> smas = [Chem.MolFromSmarts(x) for x in smas]
    >>> tform = SmartsRemover(patterns=smas)
    >>> tform.AddParent(suppl)
    >>> ms = [x for x in tform]
    >>> len(ms)
    4
    >>> Chem.MolToSmiles(ms[0])
    'C1CCC1'
    >>> Chem.MolToSmiles(ms[3])
    'NCC=O'


  """

    def __init__(self, patterns=[], wholeFragments=1, **kwargs):
        TransformNode.__init__(self, func=self.transform, **kwargs)
        self._wholeFragments = wholeFragments
        self._initPatterns(patterns)

    def _initPatterns(self, patterns):
        nPatts = len(patterns)
        targets = [None] * nPatts
        for i in range(nPatts):
            p = patterns[i]
            if type(p) in (str, bytes):
                m = Chem.MolFromSmarts(p)
                if not m:
                    raise ValueError('bad smarts: %s' % p)
                p = m
            targets[i] = p
        self._patterns = tuple(targets)

    def transform(self, cmpd):
        for patt in self._patterns:
            cmpd = Chem.DeleteSubstructs(cmpd, patt, onlyFrags=self._wholeFragments)
        return cmpd