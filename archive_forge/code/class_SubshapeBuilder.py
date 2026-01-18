import copy
import pickle
import time
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
from rdkit.Chem.Subshape import BuilderUtils, SubshapeObjects
class SubshapeBuilder(object):
    gridDims = (20, 15, 10)
    gridSpacing = 0.5
    winRad = 3.0
    nbrCount = 7
    terminalPtRadScale = 0.75
    fraction = 0.25
    stepSize = 1.0
    featFactory = None

    def SampleSubshape(self, subshape1, newSpacing):
        ogrid = subshape1.grid
        rgrid = Geometry.UniformGrid3D(self.gridDims[0], self.gridDims[1], self.gridDims[2], newSpacing)
        for idx in range(rgrid.GetSize()):
            l = rgrid.GetGridPointLoc(idx)
            v = ogrid.GetValPoint(l)
            rgrid.SetVal(idx, v)
        res = SubshapeObjects.ShapeWithSkeleton()
        res.grid = rgrid
        return res

    def GenerateSubshapeShape(self, cmpd, confId=-1, addSkeleton=True, **kwargs):
        shape = SubshapeObjects.ShapeWithSkeleton()
        shape.grid = Geometry.UniformGrid3D(self.gridDims[0], self.gridDims[1], self.gridDims[2], self.gridSpacing)
        AllChem.EncodeShape(cmpd, shape.grid, ignoreHs=False, confId=confId)
        if addSkeleton:
            conf = cmpd.GetConformer(confId)
            self.GenerateSubshapeSkeleton(shape, conf, **kwargs)
        return shape

    def __call__(self, cmpd, **kwargs):
        return self.GenerateSubshapeShape(cmpd, **kwargs)

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

    def CombineSubshapes(self, subshape1, subshape2, operation=SubshapeCombineOperations.UNION):
        cs = copy.deepcopy(subshape1)
        if operation == SubshapeCombineOperations.UNION:
            cs.grid |= subshape2.grid
        elif operation == SubshapeCombineOperations.SUM:
            cs.grid += subshape2.grid
        elif operation == SubshapeCombineOperations.INTERSECT:
            cs.grid &= subshape2.grid
        else:
            raise ValueError('bad combination operation')
        return cs