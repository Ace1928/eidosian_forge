import hashlib
import logging
import os
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Type, Union, cast
from urllib import parse
import wandb
from wandb import util
from wandb.sdk.lib import hashutil, runid
from wandb.sdk.lib.paths import LogicalPath
from ._private import MEDIA_TMP
from .base_types.media import BatchableMedia, Media
from .helper_types.bounding_boxes_2d import BoundingBoxes2D
from .helper_types.classes import Classes
from .helper_types.image_mask import ImageMask
def _set_initialization_meta(self, grouping: Optional[int]=None, caption: Optional[str]=None, classes: Optional[Union['Classes', Sequence[dict]]]=None, boxes: Optional[Union[Dict[str, 'BoundingBoxes2D'], Dict[str, dict]]]=None, masks: Optional[Union[Dict[str, 'ImageMask'], Dict[str, dict]]]=None, file_type: Optional[str]=None) -> None:
    if grouping is not None:
        self._grouping = grouping
    if caption is not None:
        self._caption = caption
    total_classes = {}
    if boxes:
        if not isinstance(boxes, dict):
            raise ValueError('Images "boxes" argument must be a dictionary')
        boxes_final: Dict[str, BoundingBoxes2D] = {}
        for key in boxes:
            box_item = boxes[key]
            if isinstance(box_item, BoundingBoxes2D):
                boxes_final[key] = box_item
            elif isinstance(box_item, dict):
                boxes_final[key] = BoundingBoxes2D(box_item, key)
            total_classes.update(boxes_final[key]._class_labels)
        self._boxes = boxes_final
    if masks:
        if not isinstance(masks, dict):
            raise ValueError('Images "masks" argument must be a dictionary')
        masks_final: Dict[str, ImageMask] = {}
        for key in masks:
            mask_item = masks[key]
            if isinstance(mask_item, ImageMask):
                masks_final[key] = mask_item
            elif isinstance(mask_item, dict):
                masks_final[key] = ImageMask(mask_item, key)
            if hasattr(masks_final[key], '_val'):
                total_classes.update(masks_final[key]._val['class_labels'])
        self._masks = masks_final
    if classes is not None:
        if isinstance(classes, Classes):
            total_classes.update({val['id']: val['name'] for val in classes._class_set})
        else:
            total_classes.update({val['id']: val['name'] for val in classes})
    if len(total_classes.keys()) > 0:
        self._classes = Classes([{'id': key, 'name': total_classes[key]} for key in total_classes.keys()])
    if self.image is not None:
        self._width, self._height = self.image.size
    self._free_ram()