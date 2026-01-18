from rdkit import RDConfig
from rdkit.ML.Descriptors import Descriptors, Parser
from rdkit.utils import chemutils
def BuildAtomDict(self):
    """ builds the local atomic dict

     We don't want to keep around all descriptor values for all atoms, so this
     method takes care of only pulling out the descriptors in which we are
     interested.

     **Notes**

       - this uses _chemutils.GetAtomicData_ to actually pull the data

    """
    self.ProcessSimpleList()
    self.ProcessCompoundList()
    self.atomDict = {}
    whereString = ' and '.join(self.nonZeroDescriptors)
    if whereString != '':
        whereString = 'where ' + whereString
    chemutils.GetAtomicData(self.atomDict, self.requiredDescriptors, self.dbName, self.dbTable, whereString, self.dbUser, self.dbPassword, includeElCounts=1)