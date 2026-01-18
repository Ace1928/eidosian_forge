import io
import pathlib
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils import BaseImageProcessor, get_size_dict
from ...image_transforms import (
from ...image_utils import (
from ...utils import (
def _update_annotation_for_padded_image(self, annotation: Dict, input_image_size: Tuple[int, int], output_image_size: Tuple[int, int], padding, update_bboxes) -> Dict:
    """
        Update the annotation for a padded image.
        """
    new_annotation = {}
    new_annotation['size'] = output_image_size
    for key, value in annotation.items():
        if key == 'masks':
            masks = value
            masks = pad(masks, padding, mode=PaddingMode.CONSTANT, constant_values=0, input_data_format=ChannelDimension.FIRST)
            masks = safe_squeeze(masks, 1)
            new_annotation['masks'] = masks
        elif key == 'boxes' and update_bboxes:
            boxes = value
            boxes *= np.asarray([input_image_size[1] / output_image_size[1], input_image_size[0] / output_image_size[0], input_image_size[1] / output_image_size[1], input_image_size[0] / output_image_size[0]])
            new_annotation['boxes'] = boxes
        elif key == 'size':
            new_annotation['size'] = output_image_size
        else:
            new_annotation[key] = value
    return new_annotation