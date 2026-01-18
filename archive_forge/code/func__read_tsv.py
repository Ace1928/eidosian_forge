import csv
import dataclasses
import json
from dataclasses import dataclass
from typing import List, Optional, Union
from ...utils import is_tf_available, is_torch_available, logging
@classmethod
def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, 'r', encoding='utf-8-sig') as f:
        return list(csv.reader(f, delimiter='\t', quotechar=quotechar))