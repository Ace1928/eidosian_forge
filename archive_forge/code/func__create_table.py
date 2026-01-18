import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import pytz
import wandb
from wandb.sdk.integration_utils.auto_logging import Response
from wandb.sdk.lib.runid import generate_id
def _create_table(self, formatted_data: List[Dict[str, Any]], model_alias: str, timestamp: float, time_elapsed: float) -> List[List[Any]]:
    """Creates a table from formatted data, model alias, timestamp, and elapsed time.

        :param formatted_data: list of dictionaries containing formatted data
        :param model_alias: alias of the model
        :param timestamp: timestamp of the data
        :param time_elapsed: time elapsed from the beginning
        :returns: list of lists, representing a table of data. [0]th element = columns. [1]st element = data
        """
    header = ['ID', 'Model Alias', 'Timestamp', 'Elapsed Time', 'Input', 'Response', 'Kwargs']
    table = [header]
    autolog_id = generate_id(length=16)
    for data in formatted_data:
        row = [autolog_id, model_alias, timestamp, time_elapsed, data['input'], data['response'], data['kwargs']]
        table.append(row)
    self.autolog_id = autolog_id
    return table