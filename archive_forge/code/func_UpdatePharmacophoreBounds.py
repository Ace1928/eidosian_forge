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
def UpdatePharmacophoreBounds(bm, atomMatch, pcophore, useDirs=False, dirLength=defaultFeatLength, mol=None):
    """ loops over a distance bounds matrix and replaces the elements
  that are altered by a pharmacophore

  **NOTE** this returns the resulting bounds matrix, but it may also
  alter the input matrix

  atomMatch is a sequence of sequences containing atom indices
  for each of the pharmacophore's features.

    >>> from rdkit import Geometry
    >>> from rdkit.Chem.Pharm3D import Pharmacophore
    >>> feats = [
    ...   ChemicalFeatures.FreeChemicalFeature('HBondAcceptor', 'HAcceptor1',
    ...                                        Geometry.Point3D(0.0, 0.0, 0.0)),
    ...   ChemicalFeatures.FreeChemicalFeature('HBondDonor', 'HDonor1',
    ...                                        Geometry.Point3D(2.65, 0.0, 0.0)),
    ...   ]
    >>> pcophore = Pharmacophore.Pharmacophore(feats)
    >>> pcophore.setLowerBound(0,1, 1.0)
    >>> pcophore.setUpperBound(0,1, 2.0)

    >>> boundsMat = numpy.array([[0.0, 3.0, 3.0],[2.0, 0.0, 3.0],[2.0, 2.0, 0.0]])
    >>> atomMatch = ((0, ), (1, ))
    >>> bm = UpdatePharmacophoreBounds(boundsMat, atomMatch, pcophore)


     In this case, there are no multi-atom features, so the result matrix
     is the same as the input:

     >>> bm is boundsMat
     True

     this means, of course, that the input boundsMat is altered:

     >>> print(', '.join([f'{x:.3f}' for x in boundsMat[0]]))
     0.000, 2.000, 3.000
     >>> print(', '.join([f'{x:.3f}' for x in boundsMat[1]]))
     1.000, 0.000, 3.000
     >>> print(', '.join([f'{x:.3f}' for x in boundsMat[2]]))
     2.000, 2.000, 0.000

  """
    replaceMap = {}
    for i, matchI in enumerate(atomMatch):
        if len(matchI) > 1:
            bm, replaceMap[i] = ReplaceGroup(matchI, bm, useDirs=useDirs)
    for i, matchI in enumerate(atomMatch):
        mi = replaceMap.get(i, matchI[0])
        for j in range(i + 1, len(atomMatch)):
            mj = replaceMap.get(j, atomMatch[j][0])
            if mi < mj:
                idx0, idx1 = (mi, mj)
            else:
                idx0, idx1 = (mj, mi)
            bm[idx0, idx1] = pcophore.getUpperBound(i, j)
            bm[idx1, idx0] = pcophore.getLowerBound(i, j)
    return bm