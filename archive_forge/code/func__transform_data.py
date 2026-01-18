import ipywidgets as widgets
from traitlets import List, Unicode, Dict, observe, Integer
from .basedatatypes import BaseFigure, BasePlotlyType
from .callbacks import BoxSelector, LassoSelector, InputDeviceState, Points
from .serializers import custom_serializers
from .version import __frontend_version__
@staticmethod
def _transform_data(to_data, from_data, should_remove=True, relayout_path=()):
    """
        Transform to_data into from_data and return relayout-style
        description of the transformation

        Parameters
        ----------
        to_data : dict|list
        from_data : dict|list

        Returns
        -------
        dict
            relayout-style description of the transformation
        """
    relayout_data = {}
    if isinstance(to_data, dict):
        if not isinstance(from_data, dict):
            raise ValueError('Mismatched data types: {to_dict} {from_data}'.format(to_dict=to_data, from_data=from_data))
        for from_prop, from_val in from_data.items():
            if isinstance(from_val, dict) or BaseFigure._is_dict_list(from_val):
                if from_prop not in to_data:
                    to_data[from_prop] = {} if isinstance(from_val, dict) else []
                input_val = to_data[from_prop]
                relayout_data.update(BaseFigureWidget._transform_data(input_val, from_val, should_remove=should_remove, relayout_path=relayout_path + (from_prop,)))
            elif from_prop not in to_data or not BasePlotlyType._vals_equal(to_data[from_prop], from_val):
                to_data[from_prop] = from_val
                relayout_path_prop = relayout_path + (from_prop,)
                relayout_data[relayout_path_prop] = from_val
        if should_remove:
            for remove_prop in set(to_data.keys()).difference(set(from_data.keys())):
                to_data.pop(remove_prop)
    elif isinstance(to_data, list):
        if not isinstance(from_data, list):
            raise ValueError('Mismatched data types: to_data: {to_data} {from_data}'.format(to_data=to_data, from_data=from_data))
        for i, from_val in enumerate(from_data):
            if i >= len(to_data):
                to_data.append(None)
            input_val = to_data[i]
            if input_val is not None and (isinstance(from_val, dict) or BaseFigure._is_dict_list(from_val)):
                relayout_data.update(BaseFigureWidget._transform_data(input_val, from_val, should_remove=should_remove, relayout_path=relayout_path + (i,)))
            elif not BasePlotlyType._vals_equal(to_data[i], from_val):
                to_data[i] = from_val
                relayout_data[relayout_path + (i,)] = from_val
    return relayout_data