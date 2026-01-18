import warnings
import json
import csv
import numpy as np
from Bio import BiopythonParserWarning
def _toOPM(plate):
    """Transform a PlateRecord object into a dictionary (PRIVATE)."""
    d = dict(plate.qualifiers.items())
    d[_csvData] = {}
    d[_csvData][_plate] = plate.id
    d[_measurements] = {}
    d[_measurements][_hour] = []
    times = set()
    for wid, w in plate._wells.items():
        d[_measurements][wid] = []
        for hour in w._signals:
            times.add(hour)
    for hour in sorted(times):
        d[_measurements][_hour].append(hour)
        for wid, w in plate._wells.items():
            if hour in w._signals:
                d[_measurements][wid].append(w[hour])
            else:
                d[_measurements][wid].append(float('nan'))
    return d