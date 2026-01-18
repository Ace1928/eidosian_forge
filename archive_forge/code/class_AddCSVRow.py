import os
import os.path as op
import nibabel as nb
import numpy as np
from math import floor, ceil
import itertools
import warnings
from .. import logging
from . import metrics as nam
from ..interfaces.base import (
from ..utils.filemanip import fname_presuffix, split_filename, ensure_list
from . import confounds
class AddCSVRow(BaseInterface):
    """
    Simple interface to add an extra row to a CSV file.

    .. note:: Requires `pandas <http://pandas.pydata.org/>`_

    .. warning:: Multi-platform thread-safe execution is possible with
        `lockfile <https://pythonhosted.org/lockfile/lockfile.html>`_. Please
        recall that (1) this module is alpha software; and (2) it should be
        installed for thread-safe writing.
        If lockfile is not installed, then the interface is not thread-safe.


    Example
    -------
    >>> from nipype.algorithms import misc
    >>> addrow = misc.AddCSVRow()
    >>> addrow.inputs.in_file = 'scores.csv'
    >>> addrow.inputs.si = 0.74
    >>> addrow.inputs.di = 0.93
    >>> addrow.inputs.subject_id = 'S400'
    >>> addrow.inputs.list_of_values = [ 0.4, 0.7, 0.3 ]
    >>> addrow.run() # doctest: +SKIP

    """
    input_spec = AddCSVRowInputSpec
    output_spec = AddCSVRowOutputSpec

    def __init__(self, infields=None, force_run=True, **kwargs):
        super(AddCSVRow, self).__init__(**kwargs)
        undefined_traits = {}
        self._infields = infields
        self._have_lock = False
        self._lock = None
        if infields:
            for key in infields:
                self.inputs.add_trait(key, traits.Any)
                self.inputs._outputs[key] = Undefined
                undefined_traits[key] = Undefined
        self.inputs.trait_set(trait_change_notify=False, **undefined_traits)
        if force_run:
            self._always_run = True

    def _run_interface(self, runtime):
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError('This interface requires pandas (http://pandas.pydata.org/) to run.') from e
        try:
            from filelock import SoftFileLock
            self._have_lock = True
        except ImportError:
            from warnings import warn
            warn('Python module filelock was not found: AddCSVRow will not be thread-safe in multi-processor execution')
        input_dict = {}
        for key, val in list(self.inputs._outputs.items()):
            if key == 'trait_added' and val in self.inputs.copyable_trait_names():
                continue
            if isinstance(val, list):
                for i, v in enumerate(val):
                    input_dict['%s_%d' % (key, i)] = v
            else:
                input_dict[key] = val
        df = pd.DataFrame([input_dict])
        if self._have_lock:
            self._lock = SoftFileLock('%s.lock' % self.inputs.in_file)
            self._lock.acquire()
        if op.exists(self.inputs.in_file):
            formerdf = pd.read_csv(self.inputs.in_file, index_col=0)
            df = pd.concat([formerdf, df], ignore_index=True)
        with open(self.inputs.in_file, 'w') as f:
            df.to_csv(f)
        if self._have_lock:
            self._lock.release()
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['csv_file'] = self.inputs.in_file
        return outputs

    def _outputs(self):
        return self._add_output_traits(super(AddCSVRow, self)._outputs())

    def _add_output_traits(self, base):
        return base