from argparse import ArgumentParser
import json
from parlai.projects.self_feeding.utils import (

    Creates .identity files from .sliced files.

    input: a .sliced file of logs (in ParlaiDialog format) from Mturk task 1, each of
        which starts with an initial prompt or topic request, and ends with a y_exp
    output: an .identity file (in self-feeding format) with y_exps used as though they
        were ys
    