from sys import version_info as _swig_python_version_info
import weakref
def RankFirst(self, index):
    """
        Ranks the index_th interval var first of all unranked interval
        vars. After that, it will no longer be considered ranked.
        """
    return _pywrapcp.SequenceVar_RankFirst(self, index)