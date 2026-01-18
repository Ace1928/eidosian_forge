import math
import numpy
from rdkit import Geometry
from rdkit.Chem.Subshape import SubshapeObjects
def AssignMolFeatsToPoints(pts, mol, featFactory, winRad):
    feats = featFactory.GetFeaturesForMol(mol)
    for i, pt in enumerate(pts):
        for feat in feats:
            if feat.GetPos().Distance(pt.location) < winRad:
                typ = feat.GetFamily()
                if typ not in pt.molFeatures:
                    pt.molFeatures.append(typ)
        print(i, pt.molFeatures)