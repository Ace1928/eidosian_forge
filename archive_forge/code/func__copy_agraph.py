import logging
from .nesting import NestedState
from .diagrams_base import BaseGraph
def _copy_agraph(graph):
    from tempfile import TemporaryFile
    with TemporaryFile() as tmp:
        if hasattr(tmp, 'file'):
            fhandle = tmp.file
        else:
            fhandle = tmp
        graph.write(fhandle)
        tmp.seek(0)
        res = graph.__class__(filename=fhandle)
        fhandle.close()
    return res