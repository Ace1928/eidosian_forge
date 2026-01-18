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
def _getFeatDict(mol, featFactory, features):
    """ **INTERNAL USE ONLY**

    >>> import os.path
    >>> from rdkit import Geometry, RDConfig, Chem
    >>> fdefFile = os.path.join(RDConfig.RDCodeDir, 'Chem/Pharm3D/test_data/BaseFeatures.fdef')
    >>> featFactory = ChemicalFeatures.BuildFeatureFactory(fdefFile)
    >>> activeFeats = [
    ...  ChemicalFeatures.FreeChemicalFeature('Acceptor', Geometry.Point3D(0.0, 0.0, 0.0)),
    ...  ChemicalFeatures.FreeChemicalFeature('Donor', Geometry.Point3D(0.0, 0.0, 0.0))]
    >>> m = Chem.MolFromSmiles('FCCN')
    >>> d = _getFeatDict(m, featFactory, activeFeats)
    >>> sorted(list(d.keys()))
    ['Acceptor', 'Donor']
    >>> donors = d['Donor']
    >>> len(donors)
    1
    >>> donors[0].GetAtomIds()
    (3,)
    >>> acceptors = d['Acceptor']
    >>> len(acceptors)
    2
    >>> acceptors[0].GetAtomIds()
    (0,)
    >>> acceptors[1].GetAtomIds()
    (3,)

  """
    molFeats = {}
    for feat in features:
        family = feat.GetFamily()
        if family not in molFeats:
            molFeats[family] = featFactory.GetFeaturesForMol(mol, includeOnly=family)
    return molFeats