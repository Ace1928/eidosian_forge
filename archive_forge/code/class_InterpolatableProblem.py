from fontTools.ttLib.ttGlyphSet import LerpGlyphSet
from fontTools.pens.basePen import AbstractPen, BasePen, DecomposingPen
from fontTools.pens.pointPen import AbstractPointPen, SegmentToPointPen
from fontTools.pens.recordingPen import RecordingPen, DecomposingRecordingPen
from fontTools.misc.transform import Transform
from collections import defaultdict, deque
from math import sqrt, copysign, atan2, pi
from enum import Enum
import itertools
import logging
class InterpolatableProblem:
    NOTHING = 'nothing'
    MISSING = 'missing'
    OPEN_PATH = 'open_path'
    PATH_COUNT = 'path_count'
    NODE_COUNT = 'node_count'
    NODE_INCOMPATIBILITY = 'node_incompatibility'
    CONTOUR_ORDER = 'contour_order'
    WRONG_START_POINT = 'wrong_start_point'
    KINK = 'kink'
    UNDERWEIGHT = 'underweight'
    OVERWEIGHT = 'overweight'
    severity = {MISSING: 1, OPEN_PATH: 2, PATH_COUNT: 3, NODE_COUNT: 4, NODE_INCOMPATIBILITY: 5, CONTOUR_ORDER: 6, WRONG_START_POINT: 7, KINK: 8, UNDERWEIGHT: 9, OVERWEIGHT: 10, NOTHING: 11}