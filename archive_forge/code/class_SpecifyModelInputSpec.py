from copy import deepcopy
import csv, math, os
from nibabel import load
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import ensure_list
from ..utils.misc import normalize_mc_params
from .. import config, logging
class SpecifyModelInputSpec(BaseInterfaceInputSpec):
    subject_info = InputMultiPath(Bunch, mandatory=True, xor=['subject_info', 'event_files', 'bids_event_file'], desc='Bunch or List(Bunch) subject-specific condition information. see :ref:`nipype.algorithms.modelgen.SpecifyModel` or for details')
    event_files = InputMultiPath(traits.List(File(exists=True)), mandatory=True, xor=['subject_info', 'event_files', 'bids_event_file'], desc='List of event description files 1, 2 or 3 column format corresponding to onsets, durations and amplitudes')
    bids_event_file = InputMultiPath(File(exists=True), mandatory=True, xor=['subject_info', 'event_files', 'bids_event_file'], desc='TSV event file containing common BIDS fields: `onset`,`duration`, and categorization and amplitude columns')
    bids_condition_column = traits.Str(default_value='trial_type', usedefault=True, desc='Column of the file passed to ``bids_event_file`` to the unique values of which events will be assignedto regressors')
    bids_amplitude_column = traits.Str(desc='Column of the file passed to ``bids_event_file`` according to which to assign amplitudes to events')
    realignment_parameters = InputMultiPath(File(exists=True), desc='Realignment parameters returned by motion correction algorithm', copyfile=False)
    parameter_source = traits.Enum('SPM', 'FSL', 'AFNI', 'FSFAST', 'NIPY', usedefault=True, desc='Source of motion parameters')
    outlier_files = InputMultiPath(File(exists=True), desc='Files containing scan outlier indices that should be tossed', copyfile=False)
    functional_runs = InputMultiPath(traits.Either(traits.List(File(exists=True)), File(exists=True)), mandatory=True, desc='Data files for model. List of 4D files or list of list of 3D files per session', copyfile=False)
    input_units = traits.Enum('secs', 'scans', mandatory=True, desc='Units of event onsets and durations (secs or scans). Output units are always in secs')
    high_pass_filter_cutoff = traits.Float(mandatory=True, desc='High-pass filter cutoff in secs')
    time_repetition = traits.Float(mandatory=True, desc='Time between the start of one volume to the start of  the next image volume.')