import json
import os
import random
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union
import regex as re
from ....file_utils import ExplicitEnum, PaddingStrategy, TensorType, add_end_docstrings, is_pandas_available
from ....tokenization_utils import AddedToken, PreTrainedTokenizer
from ....tokenization_utils_base import ENCODE_KWARGS_DOCSTRING, BatchEncoding, TextInput, TruncationStrategy
from ....utils import logging
class IndexedRowTableLinearize:
    """
    FORMAT: col: col1 | col2 | col 3 row 1 : val1 | val2 | val3 row 2 : ...
    """

    def process_table(self, table_content: Dict):
        """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        assert 'header' in table_content and 'rows' in table_content, self.PROMPT_MESSAGE
        table_str = self.process_header(table_content['header']) + ' '
        for i, row_example in enumerate(table_content['rows']):
            table_str += self.process_row(row_example, row_index=i + 1) + ' '
        return table_str.strip()

    def process_header(self, headers: List):
        """
        Given a list of headers, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        return 'col : ' + ' | '.join(headers)

    def process_row(self, row: List, row_index: int):
        """
        Given a row, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        row_str = ''
        row_cell_values = []
        for cell_value in row:
            if isinstance(cell_value, int):
                row_cell_values.append(str(cell_value))
            else:
                row_cell_values.append(cell_value)
        row_str += ' | '.join(row_cell_values)
        return 'row ' + str(row_index) + ' : ' + row_str