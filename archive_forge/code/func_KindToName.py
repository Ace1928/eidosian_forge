from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import re
from typing import Any, List, NamedTuple, Optional
from utils import bq_error
from utils import bq_id_utils
def KindToName(kind):
    """Convert a kind to just a type name."""
    return kind.partition('#')[2]