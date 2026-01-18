from __future__ import annotations
import binascii
import collections
import datetime
import enum
import glob
import io
import json
import logging
import math
import os
import re
import struct
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
import numpy
from typing import TYPE_CHECKING, BinaryIO, cast, final, overload
def _series_scn(self) -> list[TiffPageSeries] | None:
    """Return pyramidal image series in Leica SCN file."""
    from xml.etree import ElementTree as etree
    scnxml = self.pages.first.description
    root = etree.fromstring(scnxml)
    series = []
    self.pages.cache = True
    self.pages.useframes = False
    self.pages.set_keyframe(0)
    self.pages._load()
    for collection in root:
        if not collection.tag.endswith('collection'):
            continue
        for image in collection:
            if not image.tag.endswith('image'):
                continue
            name = image.attrib.get('name', 'Unknown')
            for pixels in image:
                if not pixels.tag.endswith('pixels'):
                    continue
                resolutions: dict[int, dict[str, Any]] = {}
                for dimension in pixels:
                    if not dimension.tag.endswith('dimension'):
                        continue
                    if int(image.attrib.get('sizeZ', 1)) > 1:
                        raise NotImplementedError('SCN series: Z-Stacks not supported. Please submit a sample file.')
                    sizex = int(dimension.attrib['sizeX'])
                    sizey = int(dimension.attrib['sizeY'])
                    c = int(dimension.attrib.get('c', 0))
                    z = int(dimension.attrib.get('z', 0))
                    r = int(dimension.attrib.get('r', 0))
                    ifd = int(dimension.attrib['ifd'])
                    if r in resolutions:
                        level = resolutions[r]
                        if c > level['channels']:
                            level['channels'] = c
                        if z > level['sizez']:
                            level['sizez'] = z
                        level['ifds'][c, z] = ifd
                    else:
                        resolutions[r] = {'size': [sizey, sizex], 'channels': c, 'sizez': z, 'ifds': {(c, z): ifd}}
                if not resolutions:
                    continue
                levels = []
                for r, level in sorted(resolutions.items()):
                    shape: tuple[int, ...] = (level['channels'] + 1, level['sizez'] + 1)
                    axes = 'CZ'
                    ifds: list[TiffPage | TiffFrame | None] = [None] * product(shape)
                    for (c, z), ifd in sorted(level['ifds'].items()):
                        ifds[c * shape[1] + z] = self.pages[ifd]
                    assert ifds[0] is not None
                    axes += ifds[0].axes
                    shape += ifds[0].shape
                    dtype = ifds[0].dtype
                    levels.append(TiffPageSeries(ifds, shape, dtype, axes, parent=self, name=name, kind='scn'))
                levels[0].levels.extend(levels[1:])
                series.append(levels[0])
    self.is_uniform = False
    return series