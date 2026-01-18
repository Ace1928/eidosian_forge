import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
class NominalConversor:

    def __init__(self, values):
        self.values = set(values)
        self.zero_value = values[0]

    def __call__(self, value):
        if value not in self.values:
            if value == 0:
                return self.zero_value
            raise BadNominalValue(value)
        return str(value)