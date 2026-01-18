import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
@staticmethod
def _oval_string(p1, p2, p3, p4):
    """Return /AP string defining an oval within a 4-polygon provided as points
        """

    def bezier(p, q, r):
        f = '%f %f %f %f %f %f c\n'
        return f % (p.x, p.y, q.x, q.y, r.x, r.y)
    kappa = 0.55228474983
    ml = p1 + (p4 - p1) * 0.5
    mo = p1 + (p2 - p1) * 0.5
    mr = p2 + (p3 - p2) * 0.5
    mu = p4 + (p3 - p4) * 0.5
    ol1 = ml + (p1 - ml) * kappa
    ol2 = mo + (p1 - mo) * kappa
    or1 = mo + (p2 - mo) * kappa
    or2 = mr + (p2 - mr) * kappa
    ur1 = mr + (p3 - mr) * kappa
    ur2 = mu + (p3 - mu) * kappa
    ul1 = mu + (p4 - mu) * kappa
    ul2 = ml + (p4 - ml) * kappa
    ap = '%f %f m\n' % (ml.x, ml.y)
    ap += bezier(ol1, ol2, mo)
    ap += bezier(or1, or2, mr)
    ap += bezier(ur1, ur2, mu)
    ap += bezier(ul1, ul2, ml)
    return ap