import pickle
import re
from rdkit.Chem import Descriptors as DescriptorsMod
from rdkit.ML.Descriptors import Descriptors
from rdkit.RDLogger import logger
def GetDescriptorSummaries(self):
    """ returns a tuple of summaries for the descriptors this calculator generates

    """
    res = []
    for nm in self.simpleList:
        fn = getattr(DescriptorsMod, nm, lambda x: 777)
        if hasattr(fn, '__doc__') and fn.__doc__:
            doc = fn.__doc__.split('\n\n')[0].strip()
            doc = re.sub(' *\n *', ' ', doc)
        else:
            doc = 'N/A'
        res.append(doc)
    return res