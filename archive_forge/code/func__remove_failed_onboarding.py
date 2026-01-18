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
def _remove_failed_onboarding(self):
    """
        Remove workers who failed onboarding.
        """
    df = self.dataframe
    all_workers_failing_onboarding = df.loc[df['is_onboarding'] & (df['correct'] == False), 'worker'].values
    workers_failing_onboarding = sorted(np.unique(all_workers_failing_onboarding).tolist())
    self.dataframe = df[~df['worker'].isin(workers_failing_onboarding) & ~df['is_onboarding']]