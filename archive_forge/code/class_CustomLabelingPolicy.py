from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
class CustomLabelingPolicy(LabelingPolicy):
    """ Select labels based on a user-defined policy function.

    .. warning::
        The explicit purpose of this Bokeh Model is to embed *raw JavaScript
        code* for a browser to execute. If any part of the code is derived
        from untrusted user inputs, then you must take appropriate care to
        sanitize the user input prior to passing it to Bokeh.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    args = Dict(String, AnyRef, help="\n    A mapping of names to Python objects. In particular, those can be Bokeh's models.\n    These objects are made available to the labeling policy's code snippet as the\n    values of named parameters to the callback.\n    ")
    code = String(default='', help="\n    A snippet of JavaScript code that selects a subset of labels for display.\n\n    The following arguments a are available:\n\n      * ``indices``, a set-like object containing label indices to filter\n      * ``bboxes``, an array of bounding box objects per label\n      * ``distance(i, j)``, a function computing distance (in axis dimensions)\n          between labels. If labels i and j overlap, then ``distance(i, j) <= 0``.\n      * the keys of ``args`` mapping, if any\n\n    Example:\n\n        Only display labels at even indices:\n\n        .. code-block:: javascript\n\n            code = '''\n            for (const i of indices)\n              if (i % 2 == 1)\n                indices.unset(i)\n            '''\n\n        Alternatively, as a generator:\n\n        .. code-block:: javascript\n\n            code = '''\n            for (const i of indices)\n              if (i % 2 == 0)\n                yield i\n            '''\n\n    ")