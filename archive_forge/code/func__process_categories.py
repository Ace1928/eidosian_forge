from pathlib import Path
import pandas as pd
from pandas.api.types import CategoricalDtype
from the Economic Research Service of the U.S. DEPARTMENT OF AGRICULTURE.
from http://research.stlouisfed.org/fred2.
from Eisenhower to Obama.
from V. M. Savage and G. B. West. A quantitative, theoretical
def _process_categories():
    """
    Set columns in some of the dataframes to categoricals
    """
    global diamonds, midwest, mpg, msleep
    diamonds = _ordered_categories(diamonds, {'cut': 'Fair, Good, Very Good, Premium, Ideal'.split(', '), 'clarity': 'I1 SI2 SI1 VS2 VS1 VVS2 VVS1 IF'.split(), 'color': list('DEFGHIJ')})
    mpg = _unordered_categories(mpg, 'manufacturer model trans fl drv class'.split())
    midwest = _unordered_categories(midwest, ['category'])
    msleep = _unordered_categories(msleep, ['vore', 'conservation'])