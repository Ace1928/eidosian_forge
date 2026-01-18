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
def MatchPharmacophore(matches, bounds, pcophore, useDownsampling=False, use2DLimits=False, mol=None, excludedVolumes=None, useDirs=False):
    """

  if use2DLimits is set, the molecule must also be provided and topological
  distances will also be used to filter out matches

  """
    for match, atomMatch in ConstrainedEnum(matches, mol, pcophore, bounds, use2DLimits=use2DLimits):
        bm = UpdatePharmacophoreBounds(bounds.copy(), atomMatch, pcophore, useDirs=useDirs, mol=mol)
        if excludedVolumes:
            localEvs = []
            for eV in excludedVolumes:
                featInfo = []
                for i, entry in enumerate(atomMatch):
                    info = list(eV.featInfo[i])
                    info[0] = entry
                    featInfo.append(info)
                localEvs.append(ExcludedVolume.ExcludedVolume(featInfo, eV.index, eV.exclusionDist))
            bm = AddExcludedVolumes(bm, localEvs, smoothIt=False)
        sz = bm.shape[0]
        if useDownsampling:
            indices = []
            for entry in atomMatch:
                indices.extend(entry)
            if excludedVolumes:
                for vol in localEvs:
                    indices.append(vol.index)
            bm = DownsampleBoundsMatrix(bm, indices)
        if DG.DoTriangleSmoothing(bm):
            return (0, bm, match, (sz, bm.shape[0]))
    return (1, None, None, None)