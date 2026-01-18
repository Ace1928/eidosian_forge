import numpy
from rdkit import Chem, Geometry, RDLogger
from rdkit.Chem.Subshape import SubshapeObjects
from rdkit.Numerics import Alignment
class SubshapeAligner(object):
    triangleRMSTol = 1.0
    distMetric = SubshapeDistanceMetric.PROTRUDE
    shapeDistTol = 0.2
    numFeatThresh = 3
    dirThresh = 2.6
    edgeTol = 6.0
    coarseGridToleranceMult = 1.0
    medGridToleranceMult = 1.0

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

    def _checkMatchFeatures(self, targetPts, queryPts, alignment):
        nMatched = 0
        for i in range(3):
            tgtFeats = targetPts[alignment.targetTri[i]].molFeatures
            qFeats = queryPts[alignment.queryTri[i]].molFeatures
            if not tgtFeats and (not qFeats):
                nMatched += 1
            else:
                for jFeat in tgtFeats:
                    if jFeat in qFeats:
                        nMatched += 1
                        break
            if nMatched >= self.numFeatThresh:
                break
        return nMatched >= self.numFeatThresh

    def PruneMatchesUsingFeatures(self, target, query, alignments, pruneStats=None):
        i = 0
        targetPts = target.skelPts
        queryPts = query.skelPts
        while i < len(alignments):
            alg = alignments[i]
            if not self._checkMatchFeatures(targetPts, queryPts, alg):
                if pruneStats is not None:
                    pruneStats['features'] = pruneStats.get('features', 0) + 1
                del alignments[i]
            else:
                i += 1

    def _checkMatchDirections(self, targetPts, queryPts, alignment):
        dot = 0.0
        for i in range(3):
            tgtPt = targetPts[alignment.targetTri[i]]
            queryPt = queryPts[alignment.queryTri[i]]
            qv = queryPt.shapeDirs[0]
            tv = tgtPt.shapeDirs[0]
            rotV = [0.0] * 3
            rotV[0] = alignment.transform[0, 0] * qv[0] + alignment.transform[0, 1] * qv[1] + alignment.transform[0, 2] * qv[2]
            rotV[1] = alignment.transform[1, 0] * qv[0] + alignment.transform[1, 1] * qv[1] + alignment.transform[1, 2] * qv[2]
            rotV[2] = alignment.transform[2, 0] * qv[0] + alignment.transform[2, 1] * qv[1] + alignment.transform[2, 2] * qv[2]
            dot += abs(rotV[0] * tv[0] + rotV[1] * tv[1] + rotV[2] * tv[2])
            if dot >= self.dirThresh:
                break
        alignment.dirMatch = dot
        return dot >= self.dirThresh

    def PruneMatchesUsingDirection(self, target, query, alignments, pruneStats=None):
        i = 0
        tgtPts = target.skelPts
        queryPts = query.skelPts
        while i < len(alignments):
            if not self._checkMatchDirections(tgtPts, queryPts, alignments[i]):
                if pruneStats is not None:
                    pruneStats['direction'] = pruneStats.get('direction', 0) + 1
                del alignments[i]
            else:
                i += 1

    def _addCoarseAndMediumGrids(self, mol, tgt, confId, builder):
        oSpace = builder.gridSpacing
        if mol:
            builder.gridSpacing = oSpace * 1.5
            tgt.medGrid = builder.GenerateSubshapeShape(mol, confId, addSkeleton=False)
            builder.gridSpacing = oSpace * 2
            tgt.coarseGrid = builder.GenerateSubshapeShape(mol, confId, addSkeleton=False)
            builder.gridSpacing = oSpace
        else:
            tgt.medGrid = builder.SampleSubshape(tgt, oSpace * 1.5)
            tgt.coarseGrid = builder.SampleSubshape(tgt, oSpace * 2.0)

    def _checkMatchShape(self, targetMol, target, queryMol, query, alignment, builder, targetConf, queryConf, pruneStats=None, tConfId=1001):
        matchOk = True
        TransformMol(queryMol, alignment.transform, confId=queryConf, newConfId=tConfId)
        oSpace = builder.gridSpacing
        builder.gridSpacing = oSpace * 2
        coarseGrid = builder.GenerateSubshapeShape(queryMol, tConfId, addSkeleton=False)
        d = GetShapeShapeDistance(coarseGrid, target.coarseGrid, self.distMetric)
        if d > self.shapeDistTol * self.coarseGridToleranceMult:
            matchOk = False
            if pruneStats is not None:
                pruneStats['coarseGrid'] = pruneStats.get('coarseGrid', 0) + 1
        else:
            builder.gridSpacing = oSpace * 1.5
            medGrid = builder.GenerateSubshapeShape(queryMol, tConfId, addSkeleton=False)
            d = GetShapeShapeDistance(medGrid, target.medGrid, self.distMetric)
            if d > self.shapeDistTol * self.medGridToleranceMult:
                matchOk = False
                if pruneStats is not None:
                    pruneStats['medGrid'] = pruneStats.get('medGrid', 0) + 1
            else:
                builder.gridSpacing = oSpace
                fineGrid = builder.GenerateSubshapeShape(queryMol, tConfId, addSkeleton=False)
                d = GetShapeShapeDistance(fineGrid, target, self.distMetric)
                if d > self.shapeDistTol:
                    matchOk = False
                    if pruneStats is not None:
                        pruneStats['fineGrid'] = pruneStats.get('fineGrid', 0) + 1
                alignment.shapeDist = d
        queryMol.RemoveConformer(tConfId)
        builder.gridSpacing = oSpace
        return matchOk

    def PruneMatchesUsingShape(self, targetMol, target, queryMol, query, builder, alignments, tgtConf=-1, queryConf=-1, pruneStats=None):
        if not hasattr(target, 'medGrid'):
            self._addCoarseAndMediumGrids(targetMol, target, tgtConf, builder)
        logger.info('Shape-based Pruning')
        i = 0
        nOrig = len(alignments)
        nDone = 0
        while i < len(alignments):
            alg = alignments[i]
            nDone += 1
            if not nDone % 100:
                nLeft = len(alignments)
                logger.info('  processed %d of %d. %d alignments remain' % (nDone, nOrig, nLeft))
            if not self._checkMatchShape(targetMol, target, queryMol, query, alg, builder, targetConf=tgtConf, queryConf=queryConf, pruneStats=pruneStats):
                del alignments[i]
            else:
                i += 1

    def GetSubshapeAlignments(self, targetMol, target, queryMol, query, builder, tgtConf=-1, queryConf=-1, pruneStats=None):
        import time
        if pruneStats is None:
            pruneStats = {}
        logger.info('Generating triangle matches')
        t1 = time.time()
        res = [x for x in self.GetTriangleMatches(target, query)]
        t2 = time.time()
        logger.info('Got %d possible alignments in %.1f seconds' % (len(res), t2 - t1))
        pruneStats['gtm_time'] = t2 - t1
        if builder.featFactory:
            logger.info('Doing feature pruning')
            t1 = time.time()
            self.PruneMatchesUsingFeatures(target, query, res, pruneStats=pruneStats)
            t2 = time.time()
            pruneStats['feats_time'] = t2 - t1
            logger.info('%d possible alignments remain. (%.1f seconds required)' % (len(res), t2 - t1))
        logger.info('Doing direction pruning')
        t1 = time.time()
        self.PruneMatchesUsingDirection(target, query, res, pruneStats=pruneStats)
        t2 = time.time()
        pruneStats['direction_time'] = t2 - t1
        logger.info('%d possible alignments remain. (%.1f seconds required)' % (len(res), t2 - t1))
        t1 = time.time()
        self.PruneMatchesUsingShape(targetMol, target, queryMol, query, builder, res, tgtConf=tgtConf, queryConf=queryConf, pruneStats=pruneStats)
        t2 = time.time()
        pruneStats['shape_time'] = t2 - t1
        return res

    def __call__(self, targetMol, target, queryMol, query, builder, tgtConf=-1, queryConf=-1, pruneStats=None):
        for alignment in self.GetTriangleMatches(target, query):
            if not self._checkMatchFeatures(target.skelPts, query.skelPts, alignment) and builder.featFactory:
                if pruneStats is not None:
                    pruneStats['features'] = pruneStats.get('features', 0) + 1
                continue
            if not self._checkMatchDirections(target.skelPts, query.skelPts, alignment):
                if pruneStats is not None:
                    pruneStats['direction'] = pruneStats.get('direction', 0) + 1
                continue
            if not hasattr(target, 'medGrid'):
                self._addCoarseAndMediumGrids(targetMol, target, tgtConf, builder)
            if not self._checkMatchShape(targetMol, target, queryMol, query, alignment, builder, targetConf=tgtConf, queryConf=queryConf, pruneStats=pruneStats):
                continue
            yield alignment