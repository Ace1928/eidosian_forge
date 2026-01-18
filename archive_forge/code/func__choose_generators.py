from regina import NTriangulation, writeXMLFile, readXMLFile
import tempfile
import os
from . import manifoldMethods
from . import utilities
def _choose_generators(self, *args, **kwargs):
    """
        Not necessary. We call maximalForestInDualSkeleton instead in
        _choose_generators_info.
        """
    pass