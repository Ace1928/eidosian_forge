from __future__ import division, print_function
import sys
import os
import io
import re
import glob
import math
import zlib
import time
import json
import enum
import struct
import pathlib
import warnings
import binascii
import tempfile
import datetime
import threading
import collections
import multiprocessing
import concurrent.futures
import numpy
def CZ_LSMINFO():
    return [('MagicNumber', 'u4'), ('StructureSize', 'i4'), ('DimensionX', 'i4'), ('DimensionY', 'i4'), ('DimensionZ', 'i4'), ('DimensionChannels', 'i4'), ('DimensionTime', 'i4'), ('DataType', 'i4'), ('ThumbnailX', 'i4'), ('ThumbnailY', 'i4'), ('VoxelSizeX', 'f8'), ('VoxelSizeY', 'f8'), ('VoxelSizeZ', 'f8'), ('OriginX', 'f8'), ('OriginY', 'f8'), ('OriginZ', 'f8'), ('ScanType', 'u2'), ('SpectralScan', 'u2'), ('TypeOfData', 'u4'), ('OffsetVectorOverlay', 'u4'), ('OffsetInputLut', 'u4'), ('OffsetOutputLut', 'u4'), ('OffsetChannelColors', 'u4'), ('TimeIntervall', 'f8'), ('OffsetChannelDataTypes', 'u4'), ('OffsetScanInformation', 'u4'), ('OffsetKsData', 'u4'), ('OffsetTimeStamps', 'u4'), ('OffsetEventList', 'u4'), ('OffsetRoi', 'u4'), ('OffsetBleachRoi', 'u4'), ('OffsetNextRecording', 'u4'), ('DisplayAspectX', 'f8'), ('DisplayAspectY', 'f8'), ('DisplayAspectZ', 'f8'), ('DisplayAspectTime', 'f8'), ('OffsetMeanOfRoisOverlay', 'u4'), ('OffsetTopoIsolineOverlay', 'u4'), ('OffsetTopoProfileOverlay', 'u4'), ('OffsetLinescanOverlay', 'u4'), ('ToolbarFlags', 'u4'), ('OffsetChannelWavelength', 'u4'), ('OffsetChannelFactors', 'u4'), ('ObjectiveSphereCorrection', 'f8'), ('OffsetUnmixParameters', 'u4'), ('OffsetAcquisitionParameters', 'u4'), ('OffsetCharacteristics', 'u4'), ('OffsetPalette', 'u4'), ('TimeDifferenceX', 'f8'), ('TimeDifferenceY', 'f8'), ('TimeDifferenceZ', 'f8'), ('InternalUse1', 'u4'), ('DimensionP', 'i4'), ('DimensionM', 'i4'), ('DimensionsReserved', '16i4'), ('OffsetTilePositions', 'u4'), ('', '9u4'), ('OffsetPositions', 'u4')]