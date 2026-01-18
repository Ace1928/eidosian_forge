from argparse import ArgumentParser
import json
from parlai.projects.self_feeding.utils import (
from parlai.mturk.tasks.self_feeding.rating.worlds import (

    Extracts training data for the negative response classifier (NRC) from Mturk logs.

    input: file of logs (in ParlaiDialog format) from Mturk task 1 with turn-by-turn
        quality ratings 1-5
    output: file of episodes (self-feeding format) w/ +1/-1 ratings indicating
        positive/negative example
    