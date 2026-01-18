import os
from copy import deepcopy
from nibabel import load
import numpy as np
from ... import logging
from ...utils import spm_docs as sd
from ..base import (
from ..base.traits_extension import NoDefaultSpecified
from ..matlab import MatlabCommand
from ...external.due import due, Doi, BibTeX
class SPMCommand(BaseInterface):
    """Extends `BaseInterface` class to implement SPM specific interfaces.

    WARNING: Pseudo prototype class, meant to be subclassed
    """
    input_spec = SPMCommandInputSpec
    _additional_metadata = ['field']
    _jobtype = 'basetype'
    _jobname = 'basename'
    _matlab_cmd = None
    _paths = None
    _use_mcr = None
    _references = [{'entry': BibTeX('@book{FrackowiakFristonFrithDolanMazziotta1997,author={R.S.J. Frackowiak, K.J. Friston, C.D. Frith, R.J. Dolan, and J.C. Mazziotta},title={Human Brain Function},publisher={Academic Press USA},year={1997},}'), 'description': 'The fundamental text on Statistical Parametric Mapping (SPM)', 'tags': ['implementation']}]

    def __init__(self, **inputs):
        super(SPMCommand, self).__init__(**inputs)
        self.inputs.on_trait_change(self._matlab_cmd_update, ['matlab_cmd', 'mfile', 'paths', 'use_mcr'])
        self._find_mlab_cmd_defaults()
        self._check_mlab_inputs()
        self._matlab_cmd_update()

    @classmethod
    def set_mlab_paths(cls, matlab_cmd=None, paths=None, use_mcr=None):
        cls._matlab_cmd = matlab_cmd
        cls._paths = paths
        cls._use_mcr = use_mcr
        info_dict = Info.getinfo(matlab_cmd=matlab_cmd, paths=paths, use_mcr=use_mcr)

    def _find_mlab_cmd_defaults(self):
        if self._use_mcr or 'FORCE_SPMMCR' in os.environ:
            self._use_mcr = True
            if self._matlab_cmd is None:
                try:
                    self._matlab_cmd = os.environ['SPMMCRCMD']
                except KeyError:
                    pass

    def _matlab_cmd_update(self):
        self.mlab = MatlabCommand(matlab_cmd=self.inputs.matlab_cmd, mfile=self.inputs.mfile, paths=self.inputs.paths, resource_monitor=False)
        self.mlab.inputs.script_file = 'pyscript_%s.m' % self.__class__.__name__.split('.')[-1].lower()
        if isdefined(self.inputs.use_mcr) and self.inputs.use_mcr:
            self.mlab.inputs.nodesktop = Undefined
            self.mlab.inputs.nosplash = Undefined
            self.mlab.inputs.single_comp_thread = Undefined
            self.mlab.inputs.uses_mcr = True
            self.mlab.inputs.mfile = True

    @property
    def version(self):
        info_dict = Info.getinfo(matlab_cmd=self.inputs.matlab_cmd, paths=self.inputs.paths, use_mcr=self.inputs.use_mcr)
        if info_dict:
            return '%s.%s' % (info_dict['name'].split('SPM')[-1], info_dict['release'])

    @property
    def jobtype(self):
        return self._jobtype

    @property
    def jobname(self):
        return self._jobname

    def _check_mlab_inputs(self):
        if not isdefined(self.inputs.matlab_cmd) and self._matlab_cmd:
            self.inputs.matlab_cmd = self._matlab_cmd
        if not isdefined(self.inputs.paths) and self._paths:
            self.inputs.paths = self._paths
        if not isdefined(self.inputs.use_mcr) and self._use_mcr:
            self.inputs.use_mcr = self._use_mcr

    def _run_interface(self, runtime):
        """Executes the SPM function using MATLAB."""
        self.mlab.inputs.script = self._make_matlab_command(deepcopy(self._parse_inputs()))
        results = self.mlab.run()
        runtime.returncode = results.runtime.returncode
        if self.mlab.inputs.uses_mcr:
            if 'Skipped' in results.runtime.stdout:
                self.raise_exception(runtime)
        runtime.stdout = results.runtime.stdout
        runtime.stderr = results.runtime.stderr
        runtime.merged = results.runtime.merged
        return runtime

    def _list_outputs(self):
        """Determine the expected outputs based on inputs."""
        raise NotImplementedError

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for SPM."""
        if spec.is_trait_type(traits.Bool):
            return int(val)
        elif spec.is_trait_type(traits.Tuple):
            return list(val)
        else:
            return val

    def _parse_inputs(self, skip=()):
        spmdict = {}
        metadata = dict(field=lambda t: t is not None)
        for name, spec in list(self.inputs.traits(**metadata).items()):
            if skip and name in skip:
                continue
            value = getattr(self.inputs, name)
            if not isdefined(value):
                continue
            field = spec.field
            if '.' in field:
                fields = field.split('.')
                dictref = spmdict
                for f in fields[:-1]:
                    if f not in list(dictref.keys()):
                        dictref[f] = {}
                    dictref = dictref[f]
                dictref[fields[-1]] = self._format_arg(name, spec, value)
            else:
                spmdict[field] = self._format_arg(name, spec, value)
        return [spmdict]

    def _reformat_dict_for_savemat(self, contents):
        """Encloses a dict representation within hierarchical lists.

        In order to create an appropriate SPM job structure, a Python
        dict storing the job needs to be modified so that each dict
        embedded in dict needs to be enclosed as a list element.

        Examples
        --------
        >>> a = SPMCommand()._reformat_dict_for_savemat(dict(a=1,
        ...                                                  b=dict(c=2, d=3)))
        >>> a == [{'a': 1, 'b': [{'c': 2, 'd': 3}]}]
        True

        """
        newdict = {}
        try:
            for key, value in list(contents.items()):
                if isinstance(value, dict):
                    if value:
                        newdict[key] = self._reformat_dict_for_savemat(value)
                else:
                    newdict[key] = value
            return [newdict]
        except TypeError:
            print('Requires dict input')

    def _generate_job(self, prefix='', contents=None):
        """Recursive function to generate spm job specification as a string

        Parameters
        ----------
        prefix : string
            A string that needs to get
        contents : dict
            A non-tuple Python structure containing spm job
            information gets converted to an appropriate sequence of
            matlab commands.

        """
        jobstring = ''
        if contents is None:
            return jobstring
        if isinstance(contents, list):
            for i, value in enumerate(contents):
                if prefix.endswith(')'):
                    newprefix = '%s,%d)' % (prefix[:-1], i + 1)
                else:
                    newprefix = '%s(%d)' % (prefix, i + 1)
                jobstring += self._generate_job(newprefix, value)
            return jobstring
        if isinstance(contents, dict):
            for key, value in list(contents.items()):
                newprefix = '%s.%s' % (prefix, key)
                jobstring += self._generate_job(newprefix, value)
            return jobstring
        if isinstance(contents, np.ndarray):
            if contents.dtype == np.dtype(object):
                if prefix:
                    jobstring += '%s = {...\n' % prefix
                else:
                    jobstring += '{...\n'
                for i, val in enumerate(contents):
                    if isinstance(val, np.ndarray):
                        jobstring += self._generate_job(prefix=None, contents=val)
                    elif isinstance(val, list):
                        items_format = []
                        for el in val:
                            items_format += ['{}' if not isinstance(el, (str, bytes)) else "'{}'"]
                        val_format = ', '.join(items_format).format
                        jobstring += '[{}];...\n'.format(val_format(*val))
                    elif isinstance(val, (str, bytes)):
                        jobstring += "'{}';...\n".format(val)
                    else:
                        jobstring += '%s;...\n' % str(val)
                jobstring += '};\n'
            else:
                for i, val in enumerate(contents):
                    for field in val.dtype.fields:
                        if prefix:
                            newprefix = '%s(%d).%s' % (prefix, i + 1, field)
                        else:
                            newprefix = '(%d).%s' % (i + 1, field)
                        jobstring += self._generate_job(newprefix, val[field])
            return jobstring
        if isinstance(contents, (str, bytes)):
            jobstring += "%s = '%s';\n" % (prefix, contents)
            return jobstring
        jobstring += '%s = %s;\n' % (prefix, str(contents))
        return jobstring

    def _make_matlab_command(self, contents, postscript=None):
        """Generates a mfile to build job structure
        Parameters
        ----------

        contents : list
            a list of dicts generated by _parse_inputs
            in each subclass

        cwd : string
            default os.getcwd()

        Returns
        -------
        mscript : string
            contents of a script called by matlab

        """
        cwd = os.getcwd()
        mscript = "\n        %% Generated by nipype.interfaces.spm\n        if isempty(which('spm')),\n             throw(MException('SPMCheck:NotFound', 'SPM not in matlab path'));\n        end\n        [name, version] = spm('ver');\n        fprintf('SPM version: %s Release: %s\\n',name, version);\n        fprintf('SPM path: %s\\n', which('spm'));\n        spm('Defaults','fMRI');\n\n        if strcmp(name, 'SPM8') || strcmp(name(1:5), 'SPM12'),\n           spm_jobman('initcfg');\n           spm_get_defaults('cmdline', 1);\n        end\n\n        "
        if self.mlab.inputs.mfile:
            if isdefined(self.inputs.use_v8struct) and self.inputs.use_v8struct:
                mscript += self._generate_job('jobs{1}.spm.%s.%s' % (self.jobtype, self.jobname), contents[0])
            elif self.jobname in ['st', 'smooth', 'preproc', 'preproc8', 'fmri_spec', 'fmri_est', 'factorial_design', 'defs']:
                mscript += self._generate_job('jobs{1}.%s{1}.%s(1)' % (self.jobtype, self.jobname), contents[0])
            else:
                mscript += self._generate_job('jobs{1}.%s{1}.%s{1}' % (self.jobtype, self.jobname), contents[0])
        else:
            from scipy.io import savemat
            jobdef = {'jobs': [{self.jobtype: [{self.jobname: self.reformat_dict_for_savemat(contents[0])}]}]}
            savemat(os.path.join(cwd, 'pyjobs_%s.mat' % self.jobname), jobdef)
            mscript += 'load pyjobs_%s;\n\n' % self.jobname
        mscript += "\n        spm_jobman('run', jobs);\n\n        "
        if self.inputs.use_mcr:
            mscript += "\n        if strcmp(name, 'SPM8') || strcmp(name(1:5), 'SPM12'),\n            close('all', 'force');\n        end;\n            "
        if postscript is not None:
            mscript += postscript
        return mscript