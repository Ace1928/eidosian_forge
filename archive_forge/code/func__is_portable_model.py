from . import kimpy_wrappers
from .exceptions import KIMCalculatorError
from .calculators import (
def _is_portable_model(model_name):
    """
    Returns True if the model specified is a KIM Portable Model (if it
    is not, then it must be a KIM Simulator Model -- there are no other
    types of models in KIM)
    """
    with kimpy_wrappers.ModelCollections() as col:
        model_type = col.get_item_type(model_name)
    return model_type == kimpy_wrappers.collection_item_type_portableModel