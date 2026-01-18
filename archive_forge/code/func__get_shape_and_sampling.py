import sys
import os
import struct
import logging
import numpy as np
def _get_shape_and_sampling(self):
    """Get shape and sampling without actuall using the pixel data.
        In this way, the user can get an idea what's inside without having
        to load it.
        """
    if 'NumberOfFrames' in self and self.NumberOfFrames > 1:
        if self.SamplesPerPixel > 1:
            shape = (self.SamplesPerPixel, self.NumberOfFrames, self.Rows, self.Columns)
        else:
            shape = (self.NumberOfFrames, self.Rows, self.Columns)
    elif 'SamplesPerPixel' in self:
        if self.SamplesPerPixel > 1:
            if self.BitsAllocated == 8:
                shape = (self.SamplesPerPixel, self.Rows, self.Columns)
            else:
                raise NotImplementedError('DICOM plugin only handles SamplesPerPixel > 1 if Bits Allocated = 8')
        else:
            shape = (self.Rows, self.Columns)
    else:
        raise RuntimeError('DICOM file has no SamplesPerPixel (perhaps this is a report?)')
    if 'PixelSpacing' in self:
        sampling = (float(self.PixelSpacing[0]), float(self.PixelSpacing[1]))
    else:
        sampling = (1.0, 1.0)
    if 'SliceSpacing' in self:
        sampling = (abs(self.SliceSpacing),) + sampling
    sampling = (1.0,) * (len(shape) - len(sampling)) + sampling[-len(shape):]
    self._info['shape'] = shape
    self._info['sampling'] = sampling