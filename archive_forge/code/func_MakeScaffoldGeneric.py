from rdkit import Chem
from rdkit.Chem import AllChem
def MakeScaffoldGeneric(mol):
    """ Makes a Murcko scaffold generic (i.e. all atom types->C and all bonds ->single

  >>> Chem.MolToSmiles(MakeScaffoldGeneric(Chem.MolFromSmiles('c1ccccc1')))
  'C1CCCCC1'
  >>> Chem.MolToSmiles(MakeScaffoldGeneric(Chem.MolFromSmiles('c1ncccc1')))
  'C1CCCCC1'

  The following were associated with sf.net issue 246
  >>> Chem.MolToSmiles(MakeScaffoldGeneric(Chem.MolFromSmiles('c1[nH]ccc1')))
  'C1CCCC1'
  >>> Chem.MolToSmiles(MakeScaffoldGeneric(Chem.MolFromSmiles('C1[NH2+]C1')))
  'C1CC1'
  >>> Chem.MolToSmiles(MakeScaffoldGeneric(Chem.MolFromSmiles('C1[C@](Cl)(F)O1')))
  'CC1(C)CC1'

  """
    res = Chem.Mol(mol)
    for atom in res.GetAtoms():
        if atom.GetAtomicNum() != 1:
            atom.SetAtomicNum(6)
        atom.SetIsAromatic(False)
        atom.SetIsotope(0)
        atom.SetFormalCharge(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        atom.SetNoImplicit(0)
        atom.SetNumExplicitHs(0)
    for bond in res.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    return Chem.RemoveHs(res)