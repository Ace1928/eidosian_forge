import numpy as np
import tempfile
from ...utils.misc import package_check
from ...utils.filemanip import fname_presuffix
from .base import NitimeBaseInterface
from ..base import (
class CoherenceAnalyzerInputSpec(BaseInterfaceInputSpec):
    _xor_inputs = ('in_file', 'in_TS')
    in_file = File(desc='csv file with ROIs on the columns and time-points on the rows. ROI names at the top row', exists=True, requires=('TR',))
    TR = traits.Float(desc='The TR used to collect the data in your csv file <in_file>')
    in_TS = traits.Any(desc='a nitime TimeSeries object')
    NFFT = traits.Range(low=32, value=64, usedefault=True, desc='This is the size of the window used for the spectral estimation. Use values between 32 and the number of samples in your time-series.(Defaults to 64.)')
    n_overlap = traits.Range(low=0, value=0, usedefault=True, desc='The number of samples which overlapbetween subsequent windows.(Defaults to 0)')
    frequency_range = traits.List(value=[0.02, 0.15], usedefault=True, minlen=2, maxlen=2, desc='The range of frequencies overwhich the analysis will average.[low,high] (Default [0.02,0.15]')
    output_csv_file = File(desc='File to write outputs (coherence,time-delay) with file-names: ``file_name_{coherence,timedelay}``')
    output_figure_file = File(desc='File to write output figures (coherence,time-delay) with file-names:\n``file_name_{coherence,timedelay}``. Possible formats: .png,.svg,.pdf,.jpg,...')
    figure_type = traits.Enum('matrix', 'network', usedefault=True, desc="The type of plot to generate, where 'matrix' denotes a matrix image and'network' denotes a graph representation. Default: 'matrix'")