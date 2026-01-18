import numpy
from rdkit import Chem, Geometry, RDLogger
from rdkit.Chem.Subshape import SubshapeObjects
from rdkit.Numerics import Alignment
def ClusterAlignments(mol, alignments, builder, neighborTol=0.1, distMetric=SubshapeDistanceMetric.PROTRUDE, tempConfId=1001):
    """ clusters a set of alignments and returns the cluster centroid """
    from rdkit.ML.Cluster import Butina
    dists = []
    for i in range(len(alignments)):
        TransformMol(mol, alignments[i].transform, newConfId=tempConfId)
        shapeI = builder.GenerateSubshapeShape(mol, tempConfId, addSkeleton=False)
        for j in range(i):
            TransformMol(mol, alignments[j].transform, newConfId=tempConfId + 1)
            shapeJ = builder.GenerateSubshapeShape(mol, tempConfId + 1, addSkeleton=False)
            d = GetShapeShapeDistance(shapeI, shapeJ, distMetric)
            dists.append(d)
            mol.RemoveConformer(tempConfId + 1)
        mol.RemoveConformer(tempConfId)
    clusts = Butina.ClusterData(dists, len(alignments), neighborTol, isDistData=True)
    res = [alignments[x[0]] for x in clusts]
    return res