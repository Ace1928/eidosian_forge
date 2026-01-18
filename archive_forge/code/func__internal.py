import argparse
import bisect
import os
import sys
import statistics
from typing import Dict, List, Set
from . import fitz
def _internal(args):
    fitz.message('This is from fitz.message().')
    fitz.log('This is from fitz.log().')