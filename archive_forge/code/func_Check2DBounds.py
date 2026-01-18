import math
import sys
import time
import numpy
import rdkit.DistanceGeometry as DG
from rdkit import Chem
from rdkit import RDLogger as logging
from rdkit.Chem import ChemicalFeatures, ChemicalForceFields
from rdkit.Chem import rdDistGeom as MolDG
from rdkit.Chem.Pharm3D import ExcludedVolume
from rdkit.ML.Data import Stats
def Check2DBounds(atomMatch, mol, pcophore):
    """ checks to see if a particular mapping of features onto
  a molecule satisfies a pharmacophore's 2D restrictions

    >>> from rdkit import Geometry
    >>> from rdkit.Chem.Pharm3D import Pharmacophore
    >>> activeFeats = [
    ...  ChemicalFeatures.FreeChemicalFeature('Acceptor', Geometry.Point3D(0.0, 0.0, 0.0)),
    ...  ChemicalFeatures.FreeChemicalFeature('Donor', Geometry.Point3D(0.0, 0.0, 0.0))]
    >>> pcophore= Pharmacophore.Pharmacophore(activeFeats)
    >>> pcophore.setUpperBound2D(0, 1, 3)
    >>> m = Chem.MolFromSmiles('FCC(N)CN')
    >>> Check2DBounds(((0, ), (3, )), m, pcophore)
    True
    >>> Check2DBounds(((0, ), (5, )), m, pcophore)
    False

  """
    dm = Chem.GetDistanceMatrix(mol, False, False, False)
    nFeats = len(atomMatch)
    for i in range(nFeats):
        for j in range(i + 1, nFeats):
            lowerB = pcophore._boundsMat2D[j, i]
            upperB = pcophore._boundsMat2D[i, j]
            dij = 10000
            for atomI in atomMatch[i]:
                for atomJ in atomMatch[j]:
                    try:
                        dij = min(dij, dm[atomI, atomJ])
                    except IndexError:
                        print('bad indices:', atomI, atomJ)
                        print('  shape:', dm.shape)
                        print('  match:', atomMatch)
                        print('    mol:')
                        print(Chem.MolToMolBlock(mol))
                        raise IndexError
            if dij < lowerB or dij > upperB:
                return False
    return True