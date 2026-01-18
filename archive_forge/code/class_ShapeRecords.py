from struct import pack, unpack, calcsize, error, Struct
import os
import sys
import time
import array
import tempfile
import logging
import io
from datetime import date
import zipfile
class ShapeRecords(list):
    """A class to hold a list of ShapeRecord objects. Subclasses list to ensure compatibility with
    former work and to reuse all the optimizations of the builtin list.
    In addition to the list interface, this also provides the GeoJSON __geo_interface__
    to return a FeatureCollection dictionary."""

    def __repr__(self):
        return 'ShapeRecords: {}'.format(list(self))

    @property
    def __geo_interface__(self):
        collection = {'type': 'FeatureCollection', 'features': [shaperec.__geo_interface__ for shaperec in self]}
        return collection