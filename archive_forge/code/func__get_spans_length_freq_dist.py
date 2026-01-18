import math
import sys
from collections import Counter
from pathlib import Path
from typing import (
import numpy
import srsly
import typer
from wasabi import MESSAGES, Printer, msg
from .. import util
from ..compat import Literal
from ..language import Language
from ..morphology import Morphology
from ..pipeline import Morphologizer, SpanCategorizer, TrainablePipe
from ..pipeline._edit_tree_internals.edit_trees import EditTrees
from ..pipeline._parser_internals import nonproj
from ..pipeline._parser_internals.nonproj import DELIMITER
from ..schemas import ConfigSchemaTraining
from ..training import Example, remove_bilu_prefix
from ..training.initialize import get_sourced_components
from ..util import registry, resolve_dot_names
from ..vectors import Mode as VectorsMode
from ._util import (
def _get_spans_length_freq_dist(length_dict: Dict, threshold=SPAN_LENGTH_THRESHOLD_PERCENTAGE) -> Dict[int, float]:
    """Get frequency distribution of spans length under a certain threshold"""
    all_span_lengths = []
    for _, lengths in length_dict.items():
        all_span_lengths.extend(lengths)
    freq_dist: Counter = Counter()
    for i in all_span_lengths:
        if freq_dist.get(i):
            freq_dist[i] += 1
        else:
            freq_dist[i] = 1
    freq_dist_percentage = {}
    for span_length, count in freq_dist.most_common():
        percentage = count / len(all_span_lengths) * 100.0
        percentage = round(percentage, 2)
        freq_dist_percentage[span_length] = percentage
    return freq_dist_percentage