import csv
import gzip
import json
from nltk.internals import deprecated
def _get_key_value_composed(field):
    out = field.split(HIER_SEPARATOR)
    key = out[0]
    value = HIER_SEPARATOR.join(out[1:])
    return (key, value)