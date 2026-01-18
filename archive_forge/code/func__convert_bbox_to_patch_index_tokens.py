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
def _convert_bbox_to_patch_index_tokens(self, bbox: Union[Tuple[int, int], Tuple[float, float, float, float]]) -> Tuple[str, str]:
    if len(bbox) == 2:
        idx_1, idx_2 = bbox
    else:
        num_patches_per_side = int(math.sqrt(self.num_patch_index_tokens))
        idx_1, idx_2 = coordinate_to_patch_index(bbox, num_patches_per_side)
    token_1 = f'<patch_index_{str(idx_1).zfill(4)}>'
    token_2 = f'<patch_index_{str(idx_2).zfill(4)}>'
    return (token_1, token_2)