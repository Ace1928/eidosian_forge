import matplotlib as mpl
from matplotlib.ticker import Formatter, MaxNLocator
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
def _hemisphere(self, value, value_source_crs):
    value = self._fix_lons(value)
    if value < 0:
        hemisphere = self._cardinal_labels.get('west', 'W')
    elif value > 0:
        hemisphere = self._cardinal_labels.get('east', 'E')
    else:
        hemisphere = ''
    if value == 0 and self._zero_direction_labels:
        if value_source_crs < 0:
            hemisphere = self._cardinal_labels.get('east', 'E')
        else:
            hemisphere = self._cardinal_labels.get('west', 'W')
    if value in (-180, 180) and (not self._dateline_direction_labels):
        hemisphere = ''
    return hemisphere