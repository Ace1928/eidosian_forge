import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
class ArffContainerType(TypedDict):
    description: str
    relation: str
    attributes: List
    data: Union[ArffDenseDataType, ArffSparseDataType]