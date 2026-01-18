import os
from glob import glob
import numpy as np
from ... import logging
from ...utils.filemanip import ensure_list, simplify_list, split_filename
from ..base import (
from .base import SPMCommand, SPMCommandInputSpec, scans_for_fnames, ImageFileSPM
class MultipleRegressionDesign(FactorialDesign):
    """Create SPM design for multiple regression

    Examples
    --------

    >>> mreg = MultipleRegressionDesign()
    >>> mreg.inputs.in_files = ['cont1.nii','cont2.nii']
    >>> mreg.run() # doctest: +SKIP
    """
    input_spec = MultipleRegressionDesignInputSpec

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for spm"""
        if opt in ['in_files']:
            return np.array(val, dtype=object)
        if opt in ['user_covariates']:
            outlist = []
            mapping = {'name': 'cname', 'vector': 'c', 'centering': 'iCC'}
            for dictitem in val:
                outdict = {}
                for key, keyval in list(dictitem.items()):
                    outdict[mapping[key]] = keyval
                outlist.append(outdict)
            return outlist
        return super(MultipleRegressionDesign, self)._format_arg(opt, spec, val)