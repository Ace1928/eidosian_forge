import copy
import math
import re
from typing import List, Optional, Tuple, Union
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput, is_batched
from ...processing_utils import ProcessorMixin
from ...tokenization_utils import AddedToken
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, TextInput, TruncationStrategy
from ...utils import TensorType
def _check_bboxes_for_single_text(self, bboxes):
    """
        Check `bboxes` for a single text example. It could be
            - `None`: no bounding box associated to a text.
            - A list with each element being the bounding boxes associated to one `<phrase> ... </phrase>` pair found
              in a text. This could be:
                  - `None`: no bounding box associated to a `<phrase> ... </phrase>` pair.
                  - A tuple of 2 integers: A single bounding box specified by patch indices.
                  - A tuple of 4 float point number: A single bounding box specified by (normalized) coordinates.
                  - A list containing the above 2 tuple types: Multiple bounding boxes for a
                   `<phrase> ... </phrase>` pair.
        """
    if bboxes is None:
        return
    elif not isinstance(bboxes, list):
        raise ValueError('`bboxes` (for a single text example) should be `None` or a list.')
    for bbox in bboxes:
        if bbox is None:
            continue
        elif not isinstance(bbox, list):
            bbox = [bbox]
        for element in bbox:
            if not isinstance(element, tuple) or not (len(element) == 2 and all((isinstance(x, int) for x in element)) or (len(element) == 4 and all((isinstance(x, float) for x in element)))):
                raise ValueError('Each element in `bboxes` (for a single text example) should be either `None`, a tuple containing 2 integers or 4 float point numbers, or a list containing such tuples. Also make sure the arguments `texts` and `bboxes` passed to `preprocess_text` are both in batches or both for a single example.')