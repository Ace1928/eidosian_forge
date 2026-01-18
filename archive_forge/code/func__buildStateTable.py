from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def _buildStateTable(self):
    """Return a dictionary of begin, do, end state function tuples"""
    stateTable = getattr(self.__class__, '__stateTable', None)
    if stateTable is None:
        stateTable = self.__class__.__stateTable = zipfndict(*(prefixedMethodObjDict(self, prefix) for prefix in ('begin_', 'do_', 'end_')))
    return stateTable