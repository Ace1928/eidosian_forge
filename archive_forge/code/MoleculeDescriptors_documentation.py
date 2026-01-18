import pickle
import re
from rdkit.Chem import Descriptors as DescriptorsMod
from rdkit.ML.Descriptors import Descriptors
from rdkit.RDLogger import logger
 returns a tuple of the versions of the descriptor calculators

    