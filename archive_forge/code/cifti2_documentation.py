import re
from collections import OrderedDict
from collections.abc import Iterable, MutableMapping, MutableSequence
from warnings import warn
import numpy as np
from .. import xmlutils as xml
from ..arrayproxy import reshape_dataobj
from ..caret import CaretMetaData
from ..dataobj_images import DataobjImage
from ..filebasedimages import FileBasedHeader, SerializableImage
from ..nifti1 import Nifti1Extensions
from ..nifti2 import Nifti2Header, Nifti2Image
from ..volumeutils import Recoder, make_dt_codes
Harmonize NIfTI headers with image data

        Ensures that the NIfTI-2 header records the data shape in the last three
        ``dim`` fields. Per the spec:

            Because the first four dimensions in NIfTI are reserved for space and time, the CIFTI
            dimensions are stored in the NIfTI header in dim[5] and up, where dim[5] is the length
            of the first CIFTI dimension (number of values in a row), dim[6] is the length of the
            second CIFTI dimension, and dim[7] is the length of the third CIFTI dimension, if
            applicable. The fields dim[1] through dim[4] will be 1; dim[0] will be 6 or 7,
            depending on whether a third matrix dimension exists.

        >>> import numpy as np
        >>> data = np.zeros((2,3,4))
        >>> img = Cifti2Image(data)  # doctest: +IGNORE_WARNINGS
        >>> img.shape == (2, 3, 4)
        True
        >>> img.update_headers()
        >>> img.nifti_header.get_data_shape() == (1, 1, 1, 1, 2, 3, 4)
        True
        >>> img.shape == (2, 3, 4)
        True
        