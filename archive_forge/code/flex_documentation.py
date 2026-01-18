from pathlib import Path
import param
from param.parameterized import iscoroutinefunction, resolve_ref
from ..reactive import ReactiveHTML
from .base import ListLike

        Iterates over the Viewable and any potential children in the
        applying the Selector.

        Arguments
        ---------
        selector: type or callable or None
          The selector allows selecting a subset of Viewables by
          declaring a type or callable function to filter by.

        Returns
        -------
        viewables: list(Viewable)
        