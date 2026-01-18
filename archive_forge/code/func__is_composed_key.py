import csv
import gzip
import json
from nltk.internals import deprecated
def _is_composed_key(field):
    return HIER_SEPARATOR in field