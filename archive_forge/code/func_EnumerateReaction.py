import os
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, rdChemReactions
def EnumerateReaction(reaction, bbLists, uniqueProductsOnly=False, funcGroupFilename=os.path.join(RDConfig.RDDataDir, 'Functional_Group_Hierarchy.txt'), propName='molFileValue'):
    """
  >>> testFile = os.path.join(RDConfig.RDCodeDir, 'Chem', 'SimpleEnum', 'test_data', 'boronic1.rxn')
  >>> rxn = AllChem.ReactionFromRxnFile(testFile)
  >>> rxn.Initialize()
  >>> reacts1 = ['Brc1ccccc1', 'Brc1ncccc1', 'Brc1cnccc1']
  >>> reacts1 = [Chem.MolFromSmiles(x) for x in reacts1]
  >>> reacts2 = ['CCB(O)O', 'CCCB(O)O']
  >>> reacts2 = [Chem.MolFromSmiles(x) for x in reacts2]

  >>> prods = EnumerateReaction(rxn, (reacts1, reacts2))
  >>> prods = list(prods)

  This is a bit nasty because of the symmetry of the boronic acid:

  >>> len(prods)
  12

  >>> smis = list(set([Chem.MolToSmiles(x[0]) for x in prods]))
  >>> smis.sort()
  >>> len(smis)
  6
  >>> print(smis)
  ['CCCc1ccccc1', 'CCCc1ccccn1', 'CCCc1cccnc1', 'CCc1ccccc1', 'CCc1ccccn1', 'CCc1cccnc1']

  The nastiness can be avoided at the cost of some memory by asking for only unique products:

  >>> prods = EnumerateReaction(rxn, (reacts1, reacts2), uniqueProductsOnly=True)
  >>> prods = list(prods)
  >>> len(prods)
  6
  >>> print(sorted([Chem.MolToSmiles(x[0]) for x in prods]))
  ['CCCc1ccccc1', 'CCCc1ccccn1', 'CCCc1cccnc1', 'CCc1ccccc1', 'CCc1ccccn1', 'CCc1cccnc1']


  """
    _, nError, nReacts, _, _ = PreprocessReaction(reaction)
    if nError:
        raise ValueError('bad reaction')
    if len(bbLists) != nReacts:
        raise ValueError(f'{nReacts} reactants in reaction, {len(bbLists)} bb lists supplied')

    def _uniqueOnly(lst):
        seen = []
        ps = Chem.SmilesWriteParams()
        cxflags = Chem.CXSmilesFields.CX_ENHANCEDSTEREO
        for entry in lst:
            if entry:
                smi = '.'.join(sorted([Chem.MolToCXSmiles(x, ps, cxflags) for x in entry]))
                if smi not in seen:
                    seen.append(smi)
                    yield entry
    ps = AllChem.EnumerateLibraryFromReaction(reaction, bbLists)
    if not uniqueProductsOnly:
        return ps
    return _uniqueOnly(ps)