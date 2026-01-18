import numpy
from rdkit import Chem, Geometry, RDLogger
from rdkit.Chem.Subshape import SubshapeObjects
from rdkit.Numerics import Alignment
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