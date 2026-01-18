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
def _insert_patch_index_tokens(self, text: str, bboxes: Union[List[Tuple[int]], List[Tuple[float]]]) -> str:
    if bboxes is None or len(bboxes) == 0:
        return text
    matched_phrases = list(re.finditer('<phrase>.+?</phrase>', string=text))
    if len(matched_phrases) != len(bboxes):
        raise ValueError(f'The number of elements in `bboxes` should be the same as the number of `<phrase> ... </phrase>` pairs in `text`. Got {len(matched_phrases)} v.s. {len(bboxes)} instead.')
    curr_pos = 0
    buffer = []
    for matched, bbox in zip(matched_phrases, bboxes):
        _, end = matched.span()
        buffer.append(text[curr_pos:end])
        curr_pos = end
        if bbox is None:
            continue
        if isinstance(bbox, tuple):
            bbox = [bbox]
        patch_index_strings = []
        if not all((box is not None for box in bbox)):
            raise ValueError('The multiple bounding boxes for a single phrase should not contain any `None` value.')
        for box in bbox:
            patch_index_1, patch_index_2 = self._convert_bbox_to_patch_index_tokens(box)
            patch_index_strings.append(f'{patch_index_1} {patch_index_2}')
        if len(patch_index_strings) == 0:
            continue
        position_str = ' </delimiter_of_multi_objects/> '.join(patch_index_strings)
        buffer.append(f'<object> {position_str} </object>')
    if curr_pos < len(text):
        buffer.append(text[curr_pos:])
    text = ''.join(buffer)
    return text