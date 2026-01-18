from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import dom_debugger
from . import page

    Returns a document snapshot, including the full DOM tree of the root node (including iframes,
    template contents, and imported documents) in a flattened array, as well as layout and
    white-listed computed style information for the nodes. Shadow DOM in the returned DOM tree is
    flattened.

    :param computed_styles: Whitelist of computed styles to return.
    :param include_paint_order: *(Optional)* Whether to include layout object paint orders into the snapshot.
    :param include_dom_rects: *(Optional)* Whether to include DOM rectangles (offsetRects, clientRects, scrollRects) into the snapshot
    :returns: A tuple with the following items:

        0. **documents** - The nodes in the DOM tree. The DOMNode at index 0 corresponds to the root document.
        1. **strings** - Shared string table that all string properties refer to with indexes.
    