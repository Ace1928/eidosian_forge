import pickle
import sys
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.AtomPairs import Pairs, Torsions
def GetMolFingerprint(mol, maxPathLength):
    FQuery = Chem.MolFromSmarts('F')
    CF3Query = Chem.MolFromSmarts('[$(C(F)(F)F)]')
    CF3Rxn = AllChem.ReactionFromSmarts('[*:1]-C(F)(F)F>>[*:1]-F')
    hasCF3 = mol.HasSubstructMatch(CF3Query)
    if hasCF3:
        p = CF3Rxn.RunReactants((mol,))[0][0]
        Chem.SanitizeMol(p)
        for nm in mol.GetPropNames():
            p.SetProp(nm, mol.GetProp(nm))
        mol = p
    match = mol.GetSubstructMatch(FQuery)
    fp = Torsions.GetHashedTopologicalTorsionFingerprint(mol, nBits=9192, targetSize=maxPathLength, fromAtoms=match)
    for i in range(2, maxPathLength):
        nfp = Torsions.GetHashedTopologicalTorsionFingerprint(mol, nBits=9192, targetSize=i, fromAtoms=match)
        for bit, v in nfp.GetNonzeroElements().iteritems():
            fp[bit] = fp[bit] + v
    return fp