from tempfile import NamedTemporaryFile, TemporaryDirectory
import os
import warnings
from numba.core.errors import NumbaWarning
def _repr_svg_(self):
    try:
        import graphviz
    except ImportError:
        raise RuntimeError('graphviz package needed for disasm CFG')
    jupyter_rendering = get_rendering(cmd='agfd')
    jupyter_rendering.replace('fontname="Courier",', 'fontname="Courier",fontsize=6,')
    src = graphviz.Source(jupyter_rendering)
    return src.pipe('svg').decode('UTF-8')