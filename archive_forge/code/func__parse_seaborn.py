import json
from functools import partial
from typing import Any, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
from fugue import Outputter
from fugue.extensions import namespace_candidate, parse_outputter
from ..viz._ext import Visualize
@parse_outputter.candidate(namespace_candidate('sns', lambda x: isinstance(x, str)))
def _parse_seaborn(obj: Tuple[str, str]) -> Outputter:
    return _SeabornVisualize(obj[1])