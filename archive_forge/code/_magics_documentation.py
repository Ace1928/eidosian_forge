import json
import warnings
import IPython
from IPython.core import magic_arguments
import pandas as pd
from toolz import curried
from altair.vegalite import v5 as vegalite_v5
Cell magic for displaying vega-lite visualizations in CoLab.

    %%vegalite [dataframe] [--json] [--version='v5']

    Visualize the contents of the cell using Vega-Lite, optionally
    specifying a pandas DataFrame object to be used as the dataset.

    if --json is passed, then input is parsed as json rather than yaml.
    