import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class SliceTimingInputSpec(SPMCommandInputSpec):
    in_files = InputMultiPath(traits.Either(traits.List(ImageFileSPM(exists=True)), ImageFileSPM(exists=True)), field='scans', desc='list of filenames to apply slice timing', mandatory=True, copyfile=False)
    num_slices = traits.Int(field='nslices', desc='number of slices in a volume', mandatory=True)
    time_repetition = traits.Float(field='tr', desc='time between volume acquisitions (start to start time)', mandatory=True)
    time_acquisition = traits.Float(field='ta', desc='time of volume acquisition. usually calculated as TR-(TR/num_slices)', mandatory=True)
    slice_order = traits.List(traits.Either(traits.Int(), traits.Float()), field='so', desc='1-based order or onset (in ms) in which slices are acquired', mandatory=True)
    ref_slice = traits.Either(traits.Int(), traits.Float(), field='refslice', desc='1-based Number of the reference slice or reference time point if slice_order is in onsets (ms)', mandatory=True)
    out_prefix = traits.String('a', field='prefix', usedefault=True, desc='slicetimed output prefix')