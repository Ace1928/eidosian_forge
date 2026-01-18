import numpy
from rdkit import Chem, Geometry, RDLogger
from rdkit.Chem.Subshape import SubshapeObjects
from rdkit.Numerics import Alignment
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