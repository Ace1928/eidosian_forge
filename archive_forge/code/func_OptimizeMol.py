import math
import sys
import time
import numpy
import rdkit.DistanceGeometry as DG
from rdkit import Chem
from rdkit import RDLogger as logging
from rdkit.Chem import ChemicalFeatures, ChemicalForceFields
from rdkit.Chem import rdDistGeom as MolDG
from rdkit.Chem.Pharm3D import ExcludedVolume
from rdkit.ML.Data import Stats
def OptimizeMol(mol, bm, atomMatches=None, excludedVolumes=None, forceConstant=1200.0, maxPasses=5, verbose=False):
    """  carries out a UFF optimization for a molecule optionally subject
  to the constraints in a bounds matrix

    - atomMatches, if provided, is a sequence of sequences
    - forceConstant is the force constant of the spring used to enforce
      the constraints

   returns a 2-tuple:
     1) the energy of the initial conformation
     2) the energy post-embedding
   NOTE that these energies include the energies of the constraints

    >>> from rdkit import Geometry
    >>> from rdkit.Chem.Pharm3D import Pharmacophore
    >>> m = Chem.MolFromSmiles('OCCN')
    >>> feats = [
    ...  ChemicalFeatures.FreeChemicalFeature('HBondAcceptor', 'HAcceptor1',
    ...                                       Geometry.Point3D(0.0, 0.0, 0.0)),
    ...  ChemicalFeatures.FreeChemicalFeature('HBondDonor', 'HDonor1',
    ...                                       Geometry.Point3D(2.65, 0.0, 0.0)),
    ...  ]
    >>> pcophore=Pharmacophore.Pharmacophore(feats)
    >>> pcophore.setLowerBound(0,1, 2.5)
    >>> pcophore.setUpperBound(0,1, 2.8)
    >>> atomMatch = ((0, ), (3, ))
    >>> bm, embeds, nFail = EmbedPharmacophore(m, atomMatch, pcophore, randomSeed=23, silent=1)
    >>> len(embeds)
    10
    >>> testM = embeds[0]

    Do the optimization:

    >>> e1, e2 = OptimizeMol(testM,bm,atomMatches=atomMatch)

    Optimizing should have lowered the energy:

    >>> e2 < e1
    True

    Check the constrained distance:

    >>> conf = testM.GetConformer(0)
    >>> p0 = conf.GetAtomPosition(0)
    >>> p3 = conf.GetAtomPosition(3)
    >>> d03 = p0.Distance(p3)
    >>> d03 >= pcophore.getLowerBound(0,1) - 0.01
    True
    >>> d03 <= pcophore.getUpperBound(0,1) + 0.01
    True

    If we optimize without the distance constraints (provided via the atomMatches
    argument) we're not guaranteed to get the same results, particularly in a case
    like the current one where the pharmacophore brings the atoms uncomfortably
    close together:

    >>> testM = embeds[1]
    >>> e1, e2 = OptimizeMol(testM,bm)
    >>> e2 < e1
    True
    >>> conf = testM.GetConformer(0)
    >>> p0 = conf.GetAtomPosition(0)
    >>> p3 = conf.GetAtomPosition(3)
    >>> d03 = p0.Distance(p3)
    >>> d03 >= pcophore.getLowerBound(0, 1) - 0.01
    True
    >>> d03 <= pcophore.getUpperBound(0, 1) + 0.01
    False

  """
    try:
        ff = ChemicalForceFields.UFFGetMoleculeForceField(mol)
    except Exception:
        logger.info('Problems building molecular forcefield', exc_info=True)
        return (-1.0, -1.0)
    weights = []
    if atomMatches:
        weights = [(i, j) for k in range(len(atomMatches)) for i in atomMatches[k] for l in range(k + 1, len(atomMatches)) for j in atomMatches[l]]
    for i, j in weights:
        if j < i:
            i, j = (j, i)
        ff.AddDistanceConstraint(i, j, bm[j, i], bm[i, j], forceConstant)
    if excludedVolumes:
        nAts = mol.GetNumAtoms()
        conf = mol.GetConformer()
        idx = nAts
        for exVol in excludedVolumes:
            assert exVol.pos is not None
            logger.debug(f'ff.AddExtraPoint({exVol.pos[0]:.4f},{exVol.pos[1]:.4f},{exVol.pos[2]:.4f}')
            ff.AddExtraPoint(exVol.pos[0], exVol.pos[1], exVol.pos[2], True)
            indices = []
            for localIndices, _, _ in exVol.featInfo:
                indices.extend(list(localIndices))
            indicesSet = set(indices)
            del indices
            for i in range(nAts):
                v = numpy.array(conf.GetAtomPosition(i)) - numpy.array(exVol.pos)
                d = numpy.sqrt(numpy.dot(v, v))
                if i not in indicesSet:
                    if d < 5.0:
                        logger.debug(f'ff.AddDistanceConstraint({i},{idx},{exVol.exclusionDist:.3f},1000,{forceConstant:.0f})')
                        ff.AddDistanceConstraint(i, idx, exVol.exclusionDist, 1000, forceConstant)
                else:
                    logger.debug(f'ff.AddDistanceConstraint({i},{idx},{bm[exVol.index, i]:.3f},{bm[i, exVol.index]:.3f},{forceConstant:.0f})')
                    ff.AddDistanceConstraint(i, idx, bm[exVol.index, i], bm[i, exVol.index], forceConstant)
            idx += 1
    ff.Initialize()
    e1 = ff.CalcEnergy()
    if isNaN(e1):
        raise ValueError('bogus energy')
    if verbose:
        print(Chem.MolToMolBlock(mol))
        for i, _ in enumerate(excludedVolumes):
            pos = ff.GetExtraPointPos(i)
            print('   % 7.4f   % 7.4f   % 7.4f As  0  0  0  0  0  0  0  0  0  0  0  0' % tuple(pos), file=sys.stderr)
    needsMore = ff.Minimize()
    nPasses = 0
    while needsMore and nPasses < maxPasses:
        needsMore = ff.Minimize()
        nPasses += 1
    e2 = ff.CalcEnergy()
    if isNaN(e2):
        raise ValueError('bogus energy')
    if verbose:
        print('--------')
        print(Chem.MolToMolBlock(mol))
        for i, _ in enumerate(excludedVolumes):
            pos = ff.GetExtraPointPos(i)
            print('   % 7.4f   % 7.4f   % 7.4f Sb  0  0  0  0  0  0  0  0  0  0  0  0' % tuple(pos), file=sys.stderr)
    ff = None
    return (e1, e2)