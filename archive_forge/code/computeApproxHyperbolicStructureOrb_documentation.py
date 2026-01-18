from .parseVertexGramMatrixFile import *
from .verificationError import *
from .orb import __path__ as orb_path
from snappy.snap.t3mlite import Mcomplex
import subprocess
import tempfile
import shutil
import os

    Calls Orb to compute an approximate solution to the edge equation
    for the given snappy.Triangulation.

    The result is a veriClosed.HyperbolicStructure where the edge lengths
    are in SageMath's RealDoubleField.
    