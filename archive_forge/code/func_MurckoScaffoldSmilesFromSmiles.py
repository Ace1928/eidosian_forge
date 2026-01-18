from rdkit import Chem
from rdkit.Chem import AllChem
def MurckoScaffoldSmilesFromSmiles(smiles, includeChirality=False):
    """ Returns MurckScaffold Smiles from smiles

  >>> MurckoScaffoldSmilesFromSmiles('Cc1cc(Oc2nccc(CCC)c2)ccc1')
  'c1ccc(Oc2ccccn2)cc1'

  """
    return MurckoScaffoldSmiles(smiles=smiles, includeChirality=includeChirality)