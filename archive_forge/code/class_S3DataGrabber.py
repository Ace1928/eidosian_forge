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
class S3DataGrabber(LibraryBaseInterface, IOBase):
    """
    Pull data from an Amazon S3 Bucket.

    Generic datagrabber module that wraps around glob in an
    intelligent way for neuroimaging tasks to grab files from
    Amazon S3

    Works exactly like DataGrabber, except, you must specify an
    S3 "bucket" and "bucket_path" to search for your data and a
    "local_directory" to store the data. "local_directory"
    should be a location on HDFS for Spark jobs. Additionally,
    "template" uses regex style formatting, rather than the
    glob-style found in the original DataGrabber.

    Examples
    --------
    >>> s3grab = S3DataGrabber(infields=['subj_id'], outfields=["func", "anat"])
    >>> s3grab.inputs.bucket = 'openneuro'
    >>> s3grab.inputs.sort_filelist = True
    >>> s3grab.inputs.template = '*'
    >>> s3grab.inputs.anon = True
    >>> s3grab.inputs.bucket_path = 'ds000101/ds000101_R2.0.0/uncompressed/'
    >>> s3grab.inputs.local_directory = '/tmp'
    >>> s3grab.inputs.field_template = {'anat': '%s/anat/%s_T1w.nii.gz',
    ...                                 'func': '%s/func/%s_task-simon_run-1_bold.nii.gz'}
    >>> s3grab.inputs.template_args = {'anat': [['subj_id', 'subj_id']],
    ...                                'func': [['subj_id', 'subj_id']]}
    >>> s3grab.inputs.subj_id = 'sub-01'
    >>> s3grab.run()  # doctest: +SKIP

    """
    input_spec = S3DataGrabberInputSpec
    output_spec = DynamicTraitedSpec
    _always_run = True
    _pkg = 'boto'

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
        super(S3DataGrabber, self).__init__(**kwargs)
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
        S3 specific: Downloads relevant files to a local folder specified

        Using traits.Any instead out OutputMultiPath till add_trait bug
        is fixed.
        """
        return add_traits(base, list(self.inputs.template_args.keys()))

    def _list_outputs(self):
        import boto
        if self._infields:
            for key in self._infields:
                value = getattr(self.inputs, key)
                if not isdefined(value):
                    msg = "%s requires a value for input '%s' because it was listed in 'infields'" % (self.__class__.__name__, key)
                    raise ValueError(msg)
        outputs = {}
        conn = boto.connect_s3(anon=self.inputs.anon)
        bkt = conn.get_bucket(self.inputs.bucket)
        bkt_files = list((k.key for k in bkt.list(prefix=self.inputs.bucket_path)))
        for key, args in list(self.inputs.template_args.items()):
            outputs[key] = []
            template = self.inputs.template
            if hasattr(self.inputs, 'field_template') and isdefined(self.inputs.field_template) and (key in self.inputs.field_template):
                template = self.inputs.field_template[key]
            if isdefined(self.inputs.bucket_path):
                template = os.path.join(self.inputs.bucket_path, template)
            if not args:
                filelist = []
                for fname in bkt_files:
                    if re.match(template, fname):
                        filelist.append(fname)
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
                    outfiles = []
                    for fname in bkt_files:
                        if re.match(filledtemplate, fname):
                            outfiles.append(fname)
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
            if any([val is None for val in outputs[key]]):
                outputs[key] = []
            if len(outputs[key]) == 0:
                outputs[key] = None
            elif len(outputs[key]) == 1:
                outputs[key] = outputs[key][0]
        for key, val in outputs.items():
            if isinstance(val, (list, tuple, set)):
                for i, path in enumerate(val):
                    outputs[key][i] = self.s3tolocal(path, bkt)
            else:
                outputs[key] = self.s3tolocal(val, bkt)
        return outputs

    def s3tolocal(self, s3path, bkt):
        import boto
        local_directory = str(self.inputs.local_directory)
        bucket_path = str(self.inputs.bucket_path)
        template = str(self.inputs.template)
        if not os.path.basename(local_directory) == '':
            local_directory += '/'
        if not os.path.basename(bucket_path) == '':
            bucket_path += '/'
        if template[0] == '/':
            template = template[1:]
        localpath = s3path.replace(bucket_path, local_directory)
        localdir = os.path.split(localpath)[0]
        if not os.path.exists(localdir):
            os.makedirs(localdir)
        k = boto.s3.key.Key(bkt)
        k.key = s3path
        k.get_contents_to_filename(localpath)
        return localpath