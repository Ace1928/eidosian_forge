import glob
import fnmatch
import string
import json
import os
import os.path as op
import shutil
import subprocess
import re
import copy
import tempfile
from os.path import join, dirname
from warnings import warn
from .. import config, logging
from ..utils.filemanip import (
from ..utils.misc import human_order_sorted, str2bool
from .base import (
class JSONFileSink(IOBase):
    """
    Very simple frontend for storing values into a JSON file.
    Entries already existing in in_dict will be overridden by matching
    entries dynamically added as inputs.

    .. warning::

        This is not a thread-safe node because it can write to a common
        shared location. It will not complain when it overwrites a file.

    Examples
    --------
    >>> jsonsink = JSONFileSink(input_names=['subject_id',
    ...                         'some_measurement'])
    >>> jsonsink.inputs.subject_id = 's1'
    >>> jsonsink.inputs.some_measurement = 11.4
    >>> jsonsink.run() # doctest: +SKIP

    Using a dictionary as input:

    >>> dictsink = JSONFileSink()
    >>> dictsink.inputs.in_dict = {'subject_id': 's1',
    ...                            'some_measurement': 11.4}
    >>> dictsink.run() # doctest: +SKIP

    """
    input_spec = JSONFileSinkInputSpec
    output_spec = JSONFileSinkOutputSpec

    def __init__(self, infields=[], force_run=True, **inputs):
        super(JSONFileSink, self).__init__(**inputs)
        self._input_names = infields
        undefined_traits = {}
        for key in infields:
            self.inputs.add_trait(key, traits.Any)
            self.inputs._outputs[key] = Undefined
            undefined_traits[key] = Undefined
        self.inputs.trait_set(trait_change_notify=False, **undefined_traits)
        if force_run:
            self._always_run = True

    def _process_name(self, name, val):
        if '.' in name:
            newkeys = name.split('.')
            name = newkeys.pop(0)
            nested_dict = {newkeys.pop(): val}
            for nk in reversed(newkeys):
                nested_dict = {nk: nested_dict}
            val = nested_dict
        return (name, val)

    def _list_outputs(self):
        import simplejson
        import os.path as op
        if not isdefined(self.inputs.out_file):
            out_file = op.abspath('datasink.json')
        else:
            out_file = op.abspath(self.inputs.out_file)
        out_dict = self.inputs.in_dict
        for key, val in list(self.inputs._outputs.items()):
            if not isdefined(val) or key == 'trait_added':
                continue
            key, val = self._process_name(key, val)
            out_dict[key] = val
        with open(out_file, 'w') as f:
            f.write(str(simplejson.dumps(out_dict, ensure_ascii=False)))
        outputs = self.output_spec().get()
        outputs['out_file'] = out_file
        return outputs