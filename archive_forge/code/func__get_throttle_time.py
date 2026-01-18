from collections import defaultdict
import time
import pandas as pd
from shapely.geometry import Point
import geopandas
def _get_throttle_time(provider):
    """
    Amount of time to wait between requests to a geocoding API, for providers
    that specify rate limits in their terms of service.
    """
    import geopy.geocoders
    if provider == geopy.geocoders.Nominatim:
        return 1
    else:
        return 0