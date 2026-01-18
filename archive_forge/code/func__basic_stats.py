import re
import datetime
import numpy as np
import csv
import ctypes
def _basic_stats(self, data):
    nbfac = data.size * 1.0 / (data.size - 1)
    return (np.nanmin(data), np.nanmax(data), np.mean(data), np.std(data) * nbfac)