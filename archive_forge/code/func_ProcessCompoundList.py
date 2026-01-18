from rdkit import RDConfig
from rdkit.ML.Descriptors import Descriptors, Parser
from rdkit.utils import chemutils
def ProcessCompoundList(self):
    """ Adds entries from the _compoundList_ to the list of _requiredDescriptors_

      Each compound descriptor is surveyed.  Any atomic descriptors it requires
      are added to the list of _requiredDescriptors_ to be pulled from the database.

    """
    for entry in self.compoundList:
        for atomicDesc in entry[1]:
            if atomicDesc != '' and atomicDesc not in self.requiredDescriptors:
                self.requiredDescriptors.append(atomicDesc)