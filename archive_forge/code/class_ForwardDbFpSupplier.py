import pickle
from rdkit import DataStructs
from rdkit.VLib.Node import VLibNode
class ForwardDbFpSupplier(DbFpSupplier):
    """ DbFp supplier supporting only forward iteration

    >>> from rdkit import RDConfig
    >>> from rdkit.Dbase.DbConnection import DbConnect
    >>> fName = RDConfig.RDTestDatabase
    >>> conn = DbConnect(fName,'simple_combined')
    >>> suppl = ForwardDbFpSupplier(conn.GetData())

    we can loop over the supplied fingerprints:
    
    >>> fps = []
    >>> for fp in suppl:
    ...   fps.append(fp)
    >>> len(fps)
    12

    """

    def __init__(self, *args, **kwargs):
        DbFpSupplier.__init__(self, *args, **kwargs)
        self.reset()

    def reset(self):
        DbFpSupplier.reset(self)
        self._dataIter = iter(self._data)

    def NextItem(self):
        """

          NOTE: this has side effects

        """
        d = next(self._dataIter, None)
        if d is None:
            return d
        return self._BuildFp(d)