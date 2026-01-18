import warnings
import json
import csv
import numpy as np
from Bio import BiopythonParserWarning
def _is_well(self, obj):
    """Check if the given object is a WellRecord object (PRIVATE).

        Used both for the class constructor and the __setitem__ method
        """
    if not isinstance(obj, WellRecord):
        raise ValueError(f'A WellRecord type object is needed as value (got {type(obj)})')