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
def EmbedOne(mol, name, match, pcophore, count=1, silent=0, **kwargs):
    """ generates statistics for a molecule's embeddings

  Four energies are computed for each embedding:
      1) E1: the energy (with constraints) of the initial embedding
      2) E2: the energy (with constraints) of the optimized embedding
      3) E3: the energy (no constraints) the geometry for E2
      4) E4: the energy (no constraints) of the optimized free-molecule
         (starting from the E3 geometry)

  Returns a 9-tuple:
      1) the mean value of E1
      2) the sample standard deviation of E1
      3) the mean value of E2
      4) the sample standard deviation of E2
      5) the mean value of E3
      6) the sample standard deviation of E3
      7) the mean value of E4
      8) the sample standard deviation of E4
      9) The number of embeddings that failed

  """
    global _times
    atomMatch = [list(x.GetAtomIds()) for x in match]
    bm, ms, nFailed = EmbedPharmacophore(mol, atomMatch, pcophore, count=count, silent=silent, **kwargs)
    e1s = []
    e2s = []
    e3s = []
    e4s = []
    d12s = []
    d23s = []
    d34s = []
    for m in ms:
        t1 = time.perf_counter()
        try:
            e1, e2 = OptimizeMol(m, bm, atomMatch)
        except ValueError:
            pass
        else:
            t2 = time.perf_counter()
            _times['opt1'] = _times.get('opt1', 0) + t2 - t1
            e1s.append(e1)
            e2s.append(e2)
            d12s.append(e1 - e2)
            t1 = time.perf_counter()
            try:
                e3, e4 = OptimizeMol(m, bm)
            except ValueError:
                pass
            else:
                t2 = time.perf_counter()
                _times['opt2'] = _times.get('opt2', 0) + t2 - t1
                e3s.append(e3)
                e4s.append(e4)
                d23s.append(e2 - e3)
                d34s.append(e3 - e4)
        count += 1
    try:
        e1, e1d = Stats.MeanAndDev(e1s)
    except Exception:
        e1 = -1.0
        e1d = -1.0
    try:
        e2, e2d = Stats.MeanAndDev(e2s)
    except Exception:
        e2 = -1.0
        e2d = -1.0
    try:
        e3, e3d = Stats.MeanAndDev(e3s)
    except Exception:
        e3 = -1.0
        e3d = -1.0
    try:
        e4, e4d = Stats.MeanAndDev(e4s)
    except Exception:
        e4 = -1.0
        e4d = -1.0
    if not silent:
        print(f'{name}({nFailed}): {e1:.2f}({e1d:.2f}) -> {e2:.2f}({e2d:.2f}) : {e3:.2f}({e3d:.2f}) -> {e4:.2f}({e4d:.2f})')
    return (e1, e1d, e2, e2d, e3, e3d, e4, e4d, nFailed)