import io
import os.path
import pickle
import string
from collections.abc import Iterable
from typing import Any, Callable, cast, List, Optional, Tuple, Union
from PIL import Image
from .utils import iterable_to_str, verify_str_arg
from .vision import VisionDataset
def _verify_classes(self, classes: Union[str, List[str]]) -> List[str]:
    categories = ['bedroom', 'bridge', 'church_outdoor', 'classroom', 'conference_room', 'dining_room', 'kitchen', 'living_room', 'restaurant', 'tower']
    dset_opts = ['train', 'val', 'test']
    try:
        classes = cast(str, classes)
        verify_str_arg(classes, 'classes', dset_opts)
        if classes == 'test':
            classes = [classes]
        else:
            classes = [c + '_' + classes for c in categories]
    except ValueError:
        if not isinstance(classes, Iterable):
            msg = 'Expected type str or Iterable for argument classes, but got type {}.'
            raise ValueError(msg.format(type(classes)))
        classes = list(classes)
        msg_fmtstr_type = 'Expected type str for elements in argument classes, but got type {}.'
        for c in classes:
            verify_str_arg(c, custom_msg=msg_fmtstr_type.format(type(c)))
            c_short = c.split('_')
            category, dset_opt = ('_'.join(c_short[:-1]), c_short[-1])
            msg_fmtstr = "Unknown value '{}' for {}. Valid values are {{{}}}."
            msg = msg_fmtstr.format(category, 'LSUN class', iterable_to_str(categories))
            verify_str_arg(category, valid_values=categories, custom_msg=msg)
            msg = msg_fmtstr.format(dset_opt, 'postfix', iterable_to_str(dset_opts))
            verify_str_arg(dset_opt, valid_values=dset_opts, custom_msg=msg)
    return classes