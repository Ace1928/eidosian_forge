import pickle
import re
from rdkit.Chem import Descriptors as DescriptorsMod
from rdkit.ML.Descriptors import Descriptors
from rdkit.RDLogger import logger
def GetDescriptorFuncs(self):
    """ returns a tuple of the functions used to generate this calculator's descriptors

    """
    res = []
    for nm in self.simpleList:
        fn = getattr(DescriptorsMod, nm, lambda x: 777)
        res.append(fn)
    return tuple(res)