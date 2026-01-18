import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class SurfaceSnapshotsOutputSpec(TraitedSpec):
    snapshots = OutputMultiPath(File(exists=True), desc='tiff images of the surface from different perspectives')