import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
def _get_data_object_for_encoding(matrix):
    if hasattr(matrix, 'format'):
        if matrix.format == 'coo':
            return COOData()
        else:
            raise ValueError('Cannot guess matrix format!')
    elif isinstance(matrix[0], dict):
        return LODData()
    else:
        return Data()