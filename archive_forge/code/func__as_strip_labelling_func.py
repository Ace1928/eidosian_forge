from __future__ import annotations
import typing
from abc import ABCMeta, abstractmethod
from ..exceptions import PlotnineError
def _as_strip_labelling_func(fobj: Optional[CanBeStripLabellingFunc], default: CanBeStripLabellingFunc='label_value') -> StripLabellingFunc:
    """
    Create a function that can operate on strip_label_details
    """
    if fobj is None:
        fobj = default
    if isinstance(fobj, str) and fobj in LABELLERS:
        return LABELLERS[fobj]
    if isinstance(fobj, _core_labeller):
        return fobj
    elif callable(fobj):
        if fobj.__name__ in LABELLERS:
            return fobj
        else:
            return _function_labeller(fobj)
    elif isinstance(fobj, dict):
        return _dict_labeller(fobj)
    else:
        msg = f'Could not create a labelling function for with `{fobj}`.'
        raise PlotnineError(msg)