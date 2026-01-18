import numpy
from rdkit import Chem, Geometry, RDLogger
from rdkit.Chem.Subshape import SubshapeObjects
from rdkit.Numerics import Alignment
def GetTriangleMatches(self, target, query):
    """ this is a generator function returning the possible triangle
        matches between the two shapes
    """
    ssdTol = self.triangleRMSTol ** 2 * 9
    tgtPts = target.skelPts
    queryPts = query.skelPts
    tgtLs = {}
    for i in range(len(tgtPts)):
        for j in range(i + 1, len(tgtPts)):
            l2 = (tgtPts[i].location - tgtPts[j].location).LengthSq()
            tgtLs[i, j] = l2
    queryLs = {}
    for i in range(len(queryPts)):
        for j in range(i + 1, len(queryPts)):
            l2 = (queryPts[i].location - queryPts[j].location).LengthSq()
            queryLs[i, j] = l2
    compatEdges = {}
    tol2 = self.edgeTol * self.edgeTol
    for tk, tv in tgtLs.items():
        for qk, qv in queryLs.items():
            if abs(tv - qv) < tol2:
                compatEdges[tk, qk] = 1
    seqNo = 0
    for tgtTri in _getAllTriangles(tgtPts, orderedTraversal=True):
        tgtLocs = [tgtPts[x].location for x in tgtTri]
        for queryTri in _getAllTriangles(queryPts, orderedTraversal=False):
            if ((tgtTri[0], tgtTri[1]), (queryTri[0], queryTri[1])) in compatEdges and ((tgtTri[0], tgtTri[2]), (queryTri[0], queryTri[2])) in compatEdges and (((tgtTri[1], tgtTri[2]), (queryTri[1], queryTri[2])) in compatEdges):
                queryLocs = [queryPts[x].location for x in queryTri]
                ssd, tf = Alignment.GetAlignmentTransform(tgtLocs, queryLocs)
                if ssd <= ssdTol:
                    alg = SubshapeAlignment()
                    alg.transform = tf
                    alg.triangleSSD = ssd
                    alg.targetTri = tgtTri
                    alg.queryTri = queryTri
                    alg._seqNo = seqNo
                    seqNo += 1
                    yield alg