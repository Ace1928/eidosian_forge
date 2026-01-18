import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class OneDToolPyInputSpec(AFNIPythonCommandInputSpec):
    in_file = File(desc='input file to OneDTool', argstr='-infile %s', mandatory=True, exists=True)
    set_nruns = traits.Int(desc='treat the input data as if it has nruns', argstr='-set_nruns %d')
    derivative = traits.Bool(desc='take the temporal derivative of each vector (done as first backward difference)', argstr='-derivative')
    demean = traits.Bool(desc='demean each run (new mean of each run = 0.0)', argstr='-demean')
    out_file = File(desc='write the current 1D data to FILE', argstr='-write %s', xor=['show_cormat_warnings'])
    show_censor_count = traits.Bool(desc='display the total number of censored TRs  Note : if input is a valid xmat.1D dataset, then the count will come from the header.  Otherwise the input is assumed to be a binary censorfile, and zeros are simply counted.', argstr='-show_censor_count')
    censor_motion = traits.Tuple((traits.Float(), File()), desc='Tuple of motion limit and outfile prefix. need to also set set_nruns -r set_run_lengths', argstr='-censor_motion %f %s')
    censor_prev_TR = traits.Bool(desc='for each censored TR, also censor previous', argstr='-censor_prev_TR')
    show_trs_uncensored = traits.Enum('comma', 'space', 'encoded', 'verbose', desc='display a list of TRs which were not censored in the specified style', argstr='-show_trs_uncensored %s')
    show_cormat_warnings = File(desc='Write cormat warnings to a file', argstr='-show_cormat_warnings |& tee %s', position=-1, xor=['out_file'])
    show_indices_interest = traits.Bool(desc='display column indices for regs of interest', argstr='-show_indices_interest')
    show_trs_run = traits.Int(desc='restrict -show_trs_[un]censored to the given 1-based run', argstr='-show_trs_run %d')