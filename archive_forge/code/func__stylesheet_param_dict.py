import sys
import os.path
from lxml import etree as _etree # due to validator __init__ signature
def _stylesheet_param_dict(paramsDict, kwargsDict):
    """Return a copy of paramsDict, updated with kwargsDict entries, wrapped as
    stylesheet arguments.
    kwargsDict entries with a value of None are ignored.
    """
    paramsDict = dict(paramsDict)
    for k, v in kwargsDict.items():
        if v is not None:
            paramsDict[k] = v
    paramsDict = stylesheet_params(**paramsDict)
    return paramsDict