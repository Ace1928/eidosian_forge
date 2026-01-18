import collections
import types
import numpy as np
from ..utils import (
from .base import ArgumentHandler, Dataset, Pipeline, PipelineException, build_pipeline_init_args
class TableQuestionAnsweringArgumentHandler(ArgumentHandler):
    """
    Handles arguments for the TableQuestionAnsweringPipeline
    """

    def __call__(self, table=None, query=None, **kwargs):
        requires_backends(self, 'pandas')
        import pandas as pd
        if table is None:
            raise ValueError('Keyword argument `table` cannot be None.')
        elif query is None:
            if isinstance(table, dict) and table.get('query') is not None and (table.get('table') is not None):
                tqa_pipeline_inputs = [table]
            elif isinstance(table, list) and len(table) > 0:
                if not all((isinstance(d, dict) for d in table)):
                    raise ValueError(f'Keyword argument `table` should be a list of dict, but is {(type(d) for d in table)}')
                if table[0].get('query') is not None and table[0].get('table') is not None:
                    tqa_pipeline_inputs = table
                else:
                    raise ValueError(f'If keyword argument `table` is a list of dictionaries, each dictionary should have a `table` and `query` key, but only dictionary has keys {table[0].keys()} `table` and `query` keys.')
            elif Dataset is not None and isinstance(table, Dataset) or isinstance(table, types.GeneratorType):
                return table
            else:
                raise ValueError(f'Invalid input. Keyword argument `table` should be either of type `dict` or `list`, but is {type(table)})')
        else:
            tqa_pipeline_inputs = [{'table': table, 'query': query}]
        for tqa_pipeline_input in tqa_pipeline_inputs:
            if not isinstance(tqa_pipeline_input['table'], pd.DataFrame):
                if tqa_pipeline_input['table'] is None:
                    raise ValueError('Table cannot be None.')
                tqa_pipeline_input['table'] = pd.DataFrame(tqa_pipeline_input['table'])
        return tqa_pipeline_inputs