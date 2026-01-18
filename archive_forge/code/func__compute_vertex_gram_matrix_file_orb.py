from .parseVertexGramMatrixFile import *
from .verificationError import *
from .orb import __path__ as orb_path
from snappy.snap.t3mlite import Mcomplex
import subprocess
import tempfile
import shutil
import os
def _compute_vertex_gram_matrix_file_orb(triangulation, tmpDir, verbose=False):
    """
    Calls Orb to compute an approximate solution to the edge equation
    for the given snappy.Triangulation (using directory at path tmpDir).
    """
    basename = os.path.join(tmpDir, 'manifold.tri')
    triangulation.save(basename)
    if verbose:
        print('Calling orb to find approximate solution on', basename)
    path = os.path.join(orb_path[0], 'orb_solution_for_snappea_finite_triangulation_mac')
    try:
        if verbose:
            stdout = None
        else:
            stdout = open(os.devnull, 'w')
        ret = subprocess.call([path, basename], stdout=stdout)
    except OSError as e:
        raise OrbMissingError('Seems orb binary %s is missing. Original exception: %s' % (path, e))
    if ret != 0:
        if ret <= 7:
            raise OrbSolutionTypeError('Solution type orb found was', ret)
        raise UnknownOrbFailureError('Exit code', ret)
    return basename + '.vgm'