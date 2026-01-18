import os
import re
from copy import deepcopy
import numpy as np
from .arrayproxy import ArrayProxy
from .fileslice import strided_scalar
from .spatialimages import HeaderDataError, ImageDataError, SpatialHeader, SpatialImage
from .volumeutils import Recoder

        Make `file_map` from filename `filespec`

        AFNI BRIK files can be compressed, but HEAD files cannot - see
        afni.nimh.nih.gov/pub/dist/doc/program_help/README.compression.html.
        Thus, if you have AFNI files my_image.HEAD and my_image.BRIK.gz and you
        want to load the AFNI BRIK / HEAD pair, you can specify:

            * The HEAD filename - e.g., my_image.HEAD
            * The BRIK filename w/o compressed extension - e.g., my_image.BRIK
            * The full BRIK filename - e.g., my_image.BRIK.gz

        Parameters
        ----------
        filespec : str
            Filename that might be for this image file type.

        Returns
        -------
        file_map : dict
            dict with keys ``image`` and ``header`` where values are fileholder
            objects for the respective BRIK and HEAD files

        Raises
        ------
        ImageFileError
            If `filespec` is not recognizable as being a filename for this
            image type.
        