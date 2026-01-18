import copy
import pickle
import time
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
from rdkit.Chem.Subshape import BuilderUtils, SubshapeObjects
def GenerateSubshapeSkeleton(self, shape, conf=None, terminalPtsOnly=False, skelFromConf=True):
    if conf and skelFromConf:
        pts = BuilderUtils.FindTerminalPtsFromConformer(conf, self.winRad, self.nbrCount)
    else:
        pts = BuilderUtils.FindTerminalPtsFromShape(shape, self.winRad, self.fraction)
    pts = BuilderUtils.ClusterTerminalPts(pts, self.winRad, self.terminalPtRadScale)
    BuilderUtils.ExpandTerminalPts(shape, pts, self.winRad)
    if len(pts) < 3:
        raise ValueError('only found %d terminals, need at least 3' % len(pts))
    if not terminalPtsOnly:
        pts = BuilderUtils.AppendSkeletonPoints(shape.grid, pts, self.winRad, self.stepSize)
    for pt in pts:
        BuilderUtils.CalculateDirectionsAtPoint(pt, shape.grid, self.winRad)
    if conf and self.featFactory:
        BuilderUtils.AssignMolFeatsToPoints(pts, conf.GetOwningMol(), self.featFactory, self.winRad)
    shape.skelPts = pts