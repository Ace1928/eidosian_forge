from rdkit.Chem import ChemicalFeatures
def GetDist2(self, other):
    """
    >>> from rdkit import Geometry
    >>> sfeat = ChemicalFeatures.FreeChemicalFeature('Aromatic','Foo',Geometry.Point3D(0,0,0))
    >>> fmp = FeatMapPoint()
    >>> fmp.initFromFeat(sfeat)
    >>> fmp.GetDist2(sfeat)
    0.0
    >>> sfeat.SetPos(Geometry.Point3D(2,0,0))
    >>> fmp.GetDist2(sfeat)
    4.0
    """
    return (self.GetPos() - other.GetPos()).LengthSq()