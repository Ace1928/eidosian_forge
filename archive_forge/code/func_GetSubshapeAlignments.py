import numpy
from rdkit import Chem, Geometry, RDLogger
from rdkit.Chem.Subshape import SubshapeObjects
from rdkit.Numerics import Alignment
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