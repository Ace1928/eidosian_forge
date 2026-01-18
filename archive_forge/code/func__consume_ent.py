import warnings
from typing import Dict, Iterable, Iterator, List, Tuple, Union, cast
from ..errors import Errors, Warnings
from ..tokens import Doc, Span
def _consume_ent(tags: List[str]) -> List[str]:
    if not tags:
        return []
    tag = tags.pop(0)
    target_in = 'I' + tag[1:]
    target_last = 'L' + tag[1:]
    length = 1
    while tags and tags[0] in {target_in, target_last}:
        length += 1
        tags.pop(0)
    label = tag[2:]
    if length == 1:
        if len(label) == 0:
            raise ValueError(Errors.E177.format(tag=tag))
        return ['U-' + label]
    else:
        start = 'B-' + label
        end = 'L-' + label
        middle = [f'I-{label}' for _ in range(1, length - 1)]
        return [start] + middle + [end]