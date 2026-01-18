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
class JSONFileGrabber(IOBase):
    """
    Datagrabber interface that loads a json file and generates an output for
    every first-level object

    Example
    -------

    >>> import pprint
    >>> from nipype.interfaces.io import JSONFileGrabber
    >>> jsonSource = JSONFileGrabber()
    >>> jsonSource.inputs.defaults = {'param1': 'overrideMe', 'param3': 1.0}
    >>> res = jsonSource.run()
    >>> pprint.pprint(res.outputs.get())
    {'param1': 'overrideMe', 'param3': 1.0}
    >>> jsonSource.inputs.in_file = os.path.join(datadir, 'jsongrabber.txt')
    >>> res = jsonSource.run()
    >>> pprint.pprint(res.outputs.get())  # doctest:, +ELLIPSIS
    {'param1': 'exampleStr', 'param2': 4, 'param3': 1.0}
    """
    input_spec = JSONFileGrabberInputSpec
    output_spec = DynamicTraitedSpec
    _always_run = True

    def _list_outputs(self):
        import simplejson
        outputs = {}
        if isdefined(self.inputs.in_file):
            with open(self.inputs.in_file, 'r') as f:
                data = simplejson.load(f)
            if not isinstance(data, dict):
                raise RuntimeError('JSON input has no dictionary structure')
            for key, value in list(data.items()):
                outputs[key] = value
        if isdefined(self.inputs.defaults):
            defaults = self.inputs.defaults
            for key, value in list(defaults.items()):
                if key not in list(outputs.keys()):
                    outputs[key] = value
        return outputs