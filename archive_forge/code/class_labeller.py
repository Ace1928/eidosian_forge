from __future__ import annotations
import typing
from abc import ABCMeta, abstractmethod
from ..exceptions import PlotnineError
class labeller:
    """
    Facet Strip Labelling

    When called with strip_label_details knows how to
    alter the strip labels along either dimension.

    Parameters
    ----------
    rows : str | callable
        How to label the rows
    cols : str | callable
        How to label the columns
    multi_line : bool
        Whether to place each variable on a separate line
    default : str | callable
        Fallback labelling function. If it is a string, it should be
        one of `["label_value", "label_both", "label_context"]`{.py}.
    kwargs : dict
        {variable name : function | string} pairs for
        renaming variables. A function to rename the variable
        or a string name.
    """

    def __init__(self, rows: Optional[CanBeStripLabellingFunc]=None, cols: Optional[CanBeStripLabellingFunc]=None, multi_line: bool=True, default: CanBeStripLabellingFunc='label_value', **kwargs: Callable[[str], str]):
        self.rows_labeller = _as_strip_labelling_func(rows, default)
        self.cols_labeller = _as_strip_labelling_func(cols, default)
        self.multi_line = multi_line
        self.variable_maps = kwargs

    def __call__(self, label_info: strip_label_details) -> strip_label_details:
        """
        Called to do the labelling
        """
        variable_maps = {k: v for k, v in self.variable_maps.items() if k in label_info.variables}
        if label_info.meta['dimension'] == 'rows':
            result = self.rows_labeller(label_info)
        else:
            result = self.cols_labeller(label_info)
        if variable_maps:
            d = {value: variable_maps[var] for var, value in label_info.variables.items() if var in variable_maps}
            func = _as_strip_labelling_func(d)
            result2 = func(label_info)
            result.variables.update(result2.variables)
        if not self.multi_line:
            result = result.collapse()
        return result