from rdkit import Chem
from rdkit.Chem import AllChem
def GetScaffoldForMol(mol):
    """ Return molecule object containing scaffold of mol

  >>> m = Chem.MolFromSmiles('Cc1ccccc1')
  >>> GetScaffoldForMol(m)
  <rdkit.Chem.rdchem.Mol object at 0x...>
  >>> Chem.MolToSmiles(GetScaffoldForMol(m))
  'c1ccccc1'

  >>> m = Chem.MolFromSmiles('Cc1cc(Oc2nccc(CCC)c2)ccc1')
  >>> Chem.MolToSmiles(GetScaffoldForMol(m))
  'c1ccc(Oc2ccccn2)cc1'

  """
    if 1:
        res = Chem.MurckoDecompose(mol)
        res.ClearComputedProps()
        res.UpdatePropertyCache()
        Chem.GetSymmSSSR(res)
    else:
        res = _pyGetScaffoldForMol(mol)
    return res