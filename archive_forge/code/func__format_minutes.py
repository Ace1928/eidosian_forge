import matplotlib as mpl
from matplotlib.ticker import Formatter, MaxNLocator
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
def _format_minutes(self, mn):
    """Format minutes as an integer"""
    return f'{int(mn):d}{self._minute_symbol}'