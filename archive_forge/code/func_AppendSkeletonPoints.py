import math
import numpy
from rdkit import Geometry
from rdkit.Chem.Subshape import SubshapeObjects
def AppendSkeletonPoints(shapeGrid, termPts, winRad, stepDist, maxGridVal=3, maxDistC=15.0, distTol=1.5, symFactor=1.5, verbose=False):
    nTermPts = len(termPts)
    skelPts = []
    shapeVect = shapeGrid.GetOccupancyVect()
    nGridPts = len(shapeVect)
    if verbose:
        print('generate all possible')
    for i in range(nGridPts):
        if shapeVect[i] < maxGridVal:
            continue
        posI = shapeGrid.GetGridPointLoc(i)
        ok = True
        for pt in termPts:
            dst = posI.Distance(pt.location)
            if dst < stepDist:
                ok = False
                break
        if ok:
            skelPts.append(SubshapeObjects.SkeletonPoint(location=posI))
    if verbose:
        print('Compute centroids:', len(skelPts))
    gridBoxVolume = shapeGrid.GetSpacing() ** 3
    maxVol = 4.0 * math.pi / 3.0 * winRad ** 3 * maxGridVal / gridBoxVolume
    i = 0
    while i < len(skelPts):
        pt = skelPts[i]
        count, centroid = Geometry.ComputeGridCentroid(shapeGrid, pt.location, winRad)
        centroidPtDist = centroid.Distance(pt.location)
        if centroidPtDist > maxDistC:
            del skelPts[i]
        else:
            pt.fracVol = float(count) / maxVol
            pt.location.x = centroid.x
            pt.location.y = centroid.y
            pt.location.z = centroid.z
            i += 1
    if verbose:
        print('remove points:', len(skelPts))
    res = termPts + skelPts
    i = 0
    while i < len(res):
        p = -1
        mFrac = 0.0
        ptI = res[i]
        startJ = max(i + 1, nTermPts)
        for j in range(startJ, len(res)):
            ptJ = res[j]
            distC = ptI.location.Distance(ptJ.location)
            if distC < symFactor * stepDist:
                if ptJ.fracVol > mFrac:
                    p = j
                    mFrac = ptJ.fracVol
        if p > -1:
            ptP = res.pop(p)
            j = startJ
            while j < len(res):
                ptJ = res[j]
                distC = ptI.location.Distance(ptJ.location)
                if distC < symFactor * stepDist:
                    del res[j]
                else:
                    j += 1
            res.append(ptP)
        i += 1
    return res