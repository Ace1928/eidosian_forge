import pickle
import numpy
from rdkit.ML.Data import DataUtils
def SetInputOrder(self, colNames):
    """ sets the input order

      **Arguments**

        - colNames: a list of the names of the data columns that will be passed in

      **Note**

        - you must call _SetDescriptorNames()_ first for this to work

        - if the local descriptor names do not appear in _colNames_, this will
          raise an _IndexError_ exception.
    """
    if type(colNames) != list:
        colNames = list(colNames)
    descs = [x.upper() for x in self.GetDescriptorNames()]
    self._mapOrder = [None] * len(descs)
    colNames = [x.upper() for x in colNames]
    try:
        self._mapOrder[0] = colNames.index(descs[0])
    except ValueError:
        self._mapOrder[0] = 0
    for i in range(1, len(descs) - 1):
        try:
            self._mapOrder[i] = colNames.index(descs[i])
        except ValueError:
            raise ValueError('cannot find descriptor name: %s in set %s' % (repr(descs[i]), repr(colNames)))
    try:
        self._mapOrder[-1] = colNames.index(descs[-1])
    except ValueError:
        self._mapOrder[-1] = -1