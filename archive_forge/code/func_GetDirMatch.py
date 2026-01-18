from rdkit.Chem import ChemicalFeatures
def GetDirMatch(self, other, useBest=True):
    """
    >>> from rdkit import Geometry
    >>> sfeat = ChemicalFeatures.FreeChemicalFeature('Aromatic','Foo',Geometry.Point3D(0,0,0))
    >>> fmp = FeatMapPoint()
    >>> fmp.initFromFeat(sfeat)
    >>> fmp.GetDirMatch(sfeat)
    1.0

    >>> sfeat.featDirs=[Geometry.Point3D(0,0,1),Geometry.Point3D(0,0,-1)]
    >>> fmp.featDirs=[Geometry.Point3D(0,0,1),Geometry.Point3D(1,0,0)]
    >>> fmp.GetDirMatch(sfeat)
    1.0
    >>> fmp.GetDirMatch(sfeat,useBest=True)
    1.0
    >>> fmp.GetDirMatch(sfeat,useBest=False)
    0.0

    >>> sfeat.featDirs=[Geometry.Point3D(0,0,1)]
    >>> fmp.GetDirMatch(sfeat,useBest=False)
    0.5

    >>> sfeat.featDirs=[Geometry.Point3D(0,0,1)]
    >>> fmp.featDirs=[Geometry.Point3D(0,0,-1)]
    >>> fmp.GetDirMatch(sfeat)
    -1.0
    >>> fmp.GetDirMatch(sfeat,useBest=False)
    -1.0


    """
    if not self.featDirs or not other.featDirs:
        return 1.0
    if not useBest:
        accum = 0.0
    else:
        accum = -100000.0
    for sDir in self.featDirs:
        for oDir in other.featDirs:
            d = sDir.DotProduct(oDir)
            if useBest:
                if d > accum:
                    accum = d
            else:
                accum += d
    if not useBest:
        accum /= len(self.featDirs) * len(other.featDirs)
    return accum