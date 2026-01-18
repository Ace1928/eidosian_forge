from rdkit import Chem
from rdkit.VLib.Filter import FilterNode
def _initPatterns(self, patterns, counts):
    nPatts = len(patterns)
    if len(counts) and len(counts) != nPatts:
        raise ValueError('if counts is specified, it must match patterns in length')
    if not len(counts):
        counts = [1] * nPatts
    targets = [None] * nPatts
    for i in range(nPatts):
        p = patterns[i]
        c = counts[i]
        if type(p) in (str, bytes):
            m = Chem.MolFromSmarts(p)
            if not m:
                raise ValueError('bad smarts: %s' % p)
            p = m
        targets[i] = (p, c)
    self._patterns = tuple(targets)