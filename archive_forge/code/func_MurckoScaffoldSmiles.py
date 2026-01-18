from rdkit import Chem
from rdkit.Chem import AllChem
def MurckoScaffoldSmiles(smiles=None, mol=None, includeChirality=False):
    """ Returns MurckScaffold Smiles from smiles

  >>> MurckoScaffoldSmiles('Cc1cc(Oc2nccc(CCC)c2)ccc1')
  'c1ccc(Oc2ccccn2)cc1'

  >>> MurckoScaffoldSmiles(mol=Chem.MolFromSmiles('Cc1cc(Oc2nccc(CCC)c2)ccc1'))
  'c1ccc(Oc2ccccn2)cc1'

  """
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('No molecule provided')
    scaffold = GetScaffoldForMol(mol)
    if not scaffold:
        return None
    return Chem.MolToSmiles(scaffold, includeChirality)