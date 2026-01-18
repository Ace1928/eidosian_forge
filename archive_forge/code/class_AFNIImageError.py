import os
import re
from copy import deepcopy
import numpy as np
from .arrayproxy import ArrayProxy
from .fileslice import strided_scalar
from .spatialimages import HeaderDataError, ImageDataError, SpatialHeader, SpatialImage
from .volumeutils import Recoder
class AFNIImageError(ImageDataError):
    """Error when reading AFNI BRIK files"""