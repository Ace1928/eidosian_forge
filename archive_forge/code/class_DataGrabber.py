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
class DataGrabber(IOBase):
    """
    Find files on a filesystem.

    Generic datagrabber module that wraps around glob in an
    intelligent way for neuroimaging tasks to grab files

    .. important::

       Doesn't support directories currently

    Examples
    --------
    >>> from nipype.interfaces.io import DataGrabber

    Pick all files from current directory

    >>> dg = DataGrabber()
    >>> dg.inputs.template = '*'

    Pick file foo/foo.nii from current directory

    >>> dg.inputs.template = '%s/%s.dcm'
    >>> dg.inputs.template_args['outfiles']=[['dicomdir','123456-1-1.dcm']]

    Same thing but with dynamically created fields

    >>> dg = DataGrabber(infields=['arg1','arg2'])
    >>> dg.inputs.template = '%s/%s.nii'
    >>> dg.inputs.arg1 = 'foo'
    >>> dg.inputs.arg2 = 'foo'

    however this latter form can be used with iterables and iterfield in a
    pipeline.

    Dynamically created, user-defined input and output fields

    >>> dg = DataGrabber(infields=['sid'], outfields=['func','struct','ref'])
    >>> dg.inputs.base_directory = '.'
    >>> dg.inputs.template = '%s/%s.nii'
    >>> dg.inputs.template_args['func'] = [['sid',['f3','f5']]]
    >>> dg.inputs.template_args['struct'] = [['sid',['struct']]]
    >>> dg.inputs.template_args['ref'] = [['sid','ref']]
    >>> dg.inputs.sid = 's1'

    Change the template only for output field struct. The rest use the
    general template

    >>> dg.inputs.field_template = dict(struct='%s/struct.nii')
    >>> dg.inputs.template_args['struct'] = [['sid']]

    """
    input_spec = DataGrabberInputSpec
    output_spec = DynamicTraitedSpec
    _always_run = True

    def __init__(self, infields=None, outfields=None, **kwargs):
        """
        Parameters
        ----------
        infields : list of str
            Indicates the input fields to be dynamically created

        outfields: list of str
            Indicates output fields to be dynamically created

        See class examples for usage

        """
        if not outfields:
            outfields = ['outfiles']
        super(DataGrabber, self).__init__(**kwargs)
        undefined_traits = {}
        self._infields = infields
        self._outfields = outfields
        if infields:
            for key in infields:
                self.inputs.add_trait(key, traits.Any)
                undefined_traits[key] = Undefined
        self.inputs.add_trait('field_template', traits.Dict(traits.Enum(outfields), desc='arguments that fit into template'))
        undefined_traits['field_template'] = Undefined
        if not isdefined(self.inputs.template_args):
            self.inputs.template_args = {}
        for key in outfields:
            if key not in self.inputs.template_args:
                if infields:
                    self.inputs.template_args[key] = [infields]
                else:
                    self.inputs.template_args[key] = []
        self.inputs.trait_set(trait_change_notify=False, **undefined_traits)

    def _add_output_traits(self, base):
        """

        Using traits.Any instead out OutputMultiPath till add_trait bug
        is fixed.
        """
        return add_traits(base, list(self.inputs.template_args.keys()))

    def _list_outputs(self):
        if self._infields:
            for key in self._infields:
                value = getattr(self.inputs, key)
                if not isdefined(value):
                    msg = "%s requires a value for input '%s' because it was listed in 'infields'" % (self.__class__.__name__, key)
                    raise ValueError(msg)
        outputs = {}
        for key, args in list(self.inputs.template_args.items()):
            outputs[key] = []
            template = self.inputs.template
            if hasattr(self.inputs, 'field_template') and isdefined(self.inputs.field_template) and (key in self.inputs.field_template):
                template = self.inputs.field_template[key]
            if isdefined(self.inputs.base_directory):
                template = os.path.join(os.path.abspath(self.inputs.base_directory), template)
            else:
                template = os.path.abspath(template)
            if not args:
                filelist = glob.glob(template)
                if len(filelist) == 0:
                    msg = 'Output key: %s Template: %s returned no files' % (key, template)
                    if self.inputs.raise_on_empty:
                        raise IOError(msg)
                    else:
                        warn(msg)
                else:
                    if self.inputs.sort_filelist:
                        filelist = human_order_sorted(filelist)
                    outputs[key] = simplify_list(filelist)
            for argnum, arglist in enumerate(args):
                maxlen = 1
                for arg in arglist:
                    if isinstance(arg, (str, bytes)) and hasattr(self.inputs, arg):
                        arg = getattr(self.inputs, arg)
                    if isinstance(arg, list):
                        if maxlen > 1 and len(arg) != maxlen:
                            raise ValueError('incompatible number of arguments for %s' % key)
                        if len(arg) > maxlen:
                            maxlen = len(arg)
                outfiles = []
                for i in range(maxlen):
                    argtuple = []
                    for arg in arglist:
                        if isinstance(arg, (str, bytes)) and hasattr(self.inputs, arg):
                            arg = getattr(self.inputs, arg)
                        if isinstance(arg, list):
                            argtuple.append(arg[i])
                        else:
                            argtuple.append(arg)
                    filledtemplate = template
                    if argtuple:
                        try:
                            filledtemplate = template % tuple(argtuple)
                        except TypeError as e:
                            raise TypeError(f'{e}: Template {template} failed to convert with args {tuple(argtuple)}')
                    outfiles = glob.glob(filledtemplate)
                    if len(outfiles) == 0:
                        msg = 'Output key: %s Template: %s returned no files' % (key, filledtemplate)
                        if self.inputs.raise_on_empty:
                            raise IOError(msg)
                        else:
                            warn(msg)
                        outputs[key].append(None)
                    else:
                        if self.inputs.sort_filelist:
                            outfiles = human_order_sorted(outfiles)
                        outputs[key].append(simplify_list(outfiles))
            if self.inputs.drop_blank_outputs:
                outputs[key] = [x for x in outputs[key] if x is not None]
            elif any([val is None for val in outputs[key]]):
                outputs[key] = []
            if len(outputs[key]) == 0:
                outputs[key] = None
            elif len(outputs[key]) == 1:
                outputs[key] = outputs[key][0]
        return outputs