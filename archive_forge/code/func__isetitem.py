import json
import re
import warnings
import numpy as np
import pandas as pd
import pandas.io.formats.format as fmt
def _isetitem(df, i, value):
    """Older versions of Pandas don't have df.isetitem"""
    try:
        df.isetitem(i, value)
    except AttributeError:
        df.iloc[:, i] = value