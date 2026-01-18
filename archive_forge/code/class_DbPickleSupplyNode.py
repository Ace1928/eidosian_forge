import os.path
import pickle
import sys
from rdkit import RDConfig
from rdkit.VLib.Supply import SupplyNode
class DbPickleSupplyNode(SupplyNode):
    """ Supplies pickled objects from a db result set:

  Sample Usage:
    >>> from rdkit.Dbase.DbConnection import DbConnect
  
  """

    def __init__(self, cursor, cmd, binaryCol, **kwargs):
        SupplyNode.__init__(self, **kwargs)
        self._dbResults = dbResults
        self._supplier = DbMolSupplier.RandomAccessDbMolSupplier(self._dbResults, **kwargs)

    def reset(self):
        SupplyNode.reset(self)
        self._supplier.Reset()

    def next(self):
        """

    """
        return self._supplier.next()