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
def _print_span_characteristics(span_characteristics: Dict[str, Any]):
    """Print all span characteristics into a table"""
    headers = ('Span Type', 'Length', 'SD', 'BD', 'N')
    max_col = max(30, max((len(label) for label in span_characteristics['labels'])))
    table_data = [span_characteristics['lengths'], span_characteristics['sd'], span_characteristics['bd'], span_characteristics['spans_per_type']]
    table = _format_span_row(span_data=table_data, labels=span_characteristics['labels'])
    footer_data = [span_characteristics['avg_length'], span_characteristics['avg_sd'], span_characteristics['avg_bd']]
    footer = ['Wgt. Average'] + ['{:.2f}'.format(round(f, 2)) for f in footer_data] + ['-']
    msg.table(table, footer=footer, header=headers, divider=True, aligns=['l'] + ['r'] * (len(footer_data) + 1), max_col=max_col)