from argparse import ArgumentParser
import json
from parlai.projects.self_feeding.utils import (

    Creates .stitched files from .suggested files.

    input: a .suggested file of logs (in ParlaiDialog format) from Mturk task 2, each of
        which starts with an initial prompt or topic request, and ends with a y
        that corresponds to the y_exp given in the previous turn
    output: a .stitched file (in self-feeding format) with the original mistake by the
        bot replace with the mturked y (based on y_exp)
    