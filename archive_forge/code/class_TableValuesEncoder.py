import json
import re
import warnings
import numpy as np
import pandas as pd
import pandas.io.formats.format as fmt
class TableValuesEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, (bool, int, float, str)):
            return json.JSONEncoder.default(self, obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        try:
            if obj is pd.NA:
                return str(obj)
        except AttributeError:
            pass
        if warn_on_unexpected_types:
            warnings.warn("Unexpected type '{}' for '{}'.\nYou can report this warning at https://github.com/mwouts/itables/issues\nTo silence this warning, please run:\n    import itables.options as opt\n    opt.warn_on_unexpected_types = False".format(type(obj), obj), category=RuntimeWarning)
        return str(obj)