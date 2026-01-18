from argparse import ArgumentParser
from parlai.projects.self_feeding.utils import extract_parlai_episodes
from parlai.mturk.tasks.self_feeding.rating.worlds import (

    Creates input files for y_exp mturk task from conversation/rating mturk task.

    input: file of logs (in ParlaiDialog format) from Mturk task 1 with turn-by-turn
        quality ratings 1-5
    output: file of logs (in ParlaiDialog format) sliced up to begin at the start of
        an episode or following a new topic request, and ending with a y_exp
    