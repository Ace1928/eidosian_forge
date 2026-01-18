import pickle
import re
from rdkit.Chem import Descriptors as DescriptorsMod
from rdkit.ML.Descriptors import Descriptors
from rdkit.RDLogger import logger
def GetDescriptorVersions(self):
    """ returns a tuple of the versions of the descriptor calculators

    """
    return tuple(self.descriptorVersions)