import itertools
def GetTriangles(nPts):
    """ returns a tuple with the distance indices for
     triangles composing an nPts-pharmacophore

    """
    global _trianglesInPharmacophore
    if nPts < 3:
        return []
    res = _trianglesInPharmacophore.get(nPts, [])
    if not res:
        idx1, idx2, idx3 = (0, 1, nPts - 1)
        while idx1 < nPts - 2:
            res.append((idx1, idx2, idx3))
            idx1 += 1
            idx2 += 1
            idx3 += 1
        res = tuple(res)
        _trianglesInPharmacophore[nPts] = res
    return res