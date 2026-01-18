import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
import wandb
def _infer_prediction_row_processor(example_prediction: Union[Dict, Sequence], example_input: Union[Dict, Sequence], class_labels_table: Optional['wandb.Table']=None, input_col_name: str='input', output_col_name: str='output') -> Callable:
    """Infers the composit processor for the prediction output data."""
    single_processors = {}
    if isinstance(example_prediction, dict):
        for key in example_prediction:
            key_processors = _infer_single_example_keyed_processor(example_prediction[key], class_labels_table)
            for p_key in key_processors:
                single_processors[f'{key}:{p_key}'] = _bind(lambda ndx, row, key_processor, key: key_processor(ndx, row[key], None), key_processor=key_processors[p_key], key=key)
    else:
        key = output_col_name
        key_processors = _infer_single_example_keyed_processor(example_prediction, class_labels_table, example_input if not isinstance(example_input, dict) else None)
        for p_key in key_processors:
            single_processors[f'{key}:{p_key}'] = _bind(lambda ndx, row, key_processor, key: key_processor(ndx, row[key], ndx.get_row().get('val_row').get_row().get(input_col_name) if not isinstance(example_input, dict) else None), key_processor=key_processors[p_key], key=key)

    def processor(ndx, row):
        return {key: single_processors[key](ndx, row) for key in single_processors}
    return processor