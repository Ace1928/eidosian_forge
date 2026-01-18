import re
from rdkit import Geometry
from rdkit.Chem.FeatMaps import FeatMapPoint, FeatMaps


ScoreMode=All
DirScoreMode=Ignore

BeginParams
  family=Aromatic radius=2.5 width=1.0 profile=Gaussian
  family=Acceptor radius=1.5
EndParams

# optional
BeginPoints
  family=Acceptor pos=(1.0, 0.0, 5.0) weight=1.25 dir=(1, 1, 0)
  family=Aromatic pos=(0.0,1.0,0.0) weight=2.0 dir=(0,0,1) dir=(0,0,-1)
  family=Acceptor pos=(1.0,1.0,2.0) weight=1.25
EndPoints

