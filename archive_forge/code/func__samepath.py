import os
import argparse
import logging
import shutil
import multiprocessing as mp
from contextlib import closing
from functools import partial
import fontTools
from .ufo import font_to_quadratic, fonts_to_quadratic
def _samepath(path1, path2):
    path1 = os.path.normcase(os.path.abspath(os.path.realpath(path1)))
    path2 = os.path.normcase(os.path.abspath(os.path.realpath(path2)))
    return path1 == path2