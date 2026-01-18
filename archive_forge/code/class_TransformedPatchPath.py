import copy
import functools
import textwrap
import weakref
import math
import numpy as np
from numpy.linalg import inv
from matplotlib import _api
from matplotlib._path import (
from .path import Path
class TransformedPatchPath(TransformedPath):
    """
    A `TransformedPatchPath` caches a non-affine transformed copy of the
    `~.patches.Patch`. This cached copy is automatically updated when the
    non-affine part of the transform or the patch changes.
    """

    def __init__(self, patch):
        """
        Parameters
        ----------
        patch : `~.patches.Patch`
        """
        super().__init__(patch.get_path(), patch.get_transform())
        self._patch = patch

    def _revalidate(self):
        patch_path = self._patch.get_path()
        if patch_path != self._path:
            self._path = patch_path
            self._transformed_path = None
        super()._revalidate()