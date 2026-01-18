import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from scipy.stats import binom_test
from parlai.mturk.core.mturk_data_handler import MTurkDataHandler
from parlai import __file__ as parlai_filepath
from parlai.core.params import ParlaiParser
import json
from IPython.core.display import HTML
def _extract_to_dataframe(self) -> pd.DataFrame:
    """
        Extract the data from the run to a pandas dataframe.
        """
    logger = MTurkDataHandler(file_name=self.db_path)
    hits = logger.get_pairings_for_run(self.run_id)
    dataframe: List[Dict[str, Any]] = []
    for hit in hits:
        if hit['conversation_id'] is None:
            continue
        data = self._get_hit_data(hit, logger)
        if data is None:
            continue
        for r_idx, task_data in enumerate(data['task_data']):
            response_data = data['response']['task_data'][r_idx]
            if response_data is None or not response_data:
                continue
            response = self._extract_response_data(data, task_data, hit, response_data)
            dataframe.append(response)
    return pd.DataFrame(dataframe)