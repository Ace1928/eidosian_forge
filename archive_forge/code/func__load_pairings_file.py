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
def _load_pairings_file(self):
    """
        Load the pairings file, if provided.

        Allows for visualization of the conversations turkers rated.
        """
    df = self.dataframe
    if not self.pairings_filepath or not os.path.exists(self.pairings_filepath):
        return
    pairings = []
    with open(self.pairings_filepath, 'r') as f:
        for line in f:
            pair = json.loads(line)
            model1, model2 = pair['speakers_to_eval']
            pair[model1] = pair['dialogue_dicts'][0]
            pair[model2] = pair['dialogue_dicts'][1]
            del pair['dialogue_dicts']
            pairings.append(pair)
    pairs_to_eval = [pairings[i] for i in df.pairing_id.values.tolist()]
    winner_dialogues = []
    loser_dialogues = []
    for i, (_, row) in enumerate(df.iterrows()):
        winner = row['winner']
        loser = row['loser']
        winner_dialogues.append(pairs_to_eval[i][winner])
        loser_dialogues.append(pairs_to_eval[i][loser])
    df.loc[:, 'pairs_to_eval'] = pd.Series(pairs_to_eval, index=df.index)
    df.loc[:, 'winner_dialogue'] = pd.Series(winner_dialogues, index=df.index)
    df.loc[:, 'loser_dialogue'] = pd.Series(loser_dialogues, index=df.index)
    self.dataframe = df