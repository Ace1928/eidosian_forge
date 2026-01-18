import numpy as np
from ... import Point, PolyLineROI
from ... import functions as pgfn
from ... import metaarray as metaarray
from . import functions
from .common import CtrlNode, PlottingCtrlNode, metaArrayWrapper
Return a list of Point() where the x position is set to the nearest x value in *data* for each point in *pts*.