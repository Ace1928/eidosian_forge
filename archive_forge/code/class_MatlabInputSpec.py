import os
from .. import config
from .base import (
class MatlabInputSpec(CommandLineInputSpec):
    """Basic expected inputs to Matlab interface"""
    script = traits.Str(argstr='-r "%s;exit"', desc='m-code to run', mandatory=True, position=-1)
    uses_mcr = traits.Bool(desc='use MCR interface', xor=['nodesktop', 'nosplash', 'single_comp_thread'], nohash=True)
    nodesktop = traits.Bool(True, argstr='-nodesktop', usedefault=True, desc='Switch off desktop mode on unix platforms', nohash=True)
    nosplash = traits.Bool(True, argstr='-nosplash', usedefault=True, desc='Switch of splash screen', nohash=True)
    logfile = File(argstr='-logfile %s', desc='Save matlab output to log')
    single_comp_thread = traits.Bool(argstr='-singleCompThread', desc='force single threaded operation', nohash=True)
    mfile = traits.Bool(True, desc='Run m-code using m-file', usedefault=True)
    script_file = File('pyscript.m', usedefault=True, desc='Name of file to write m-code to')
    paths = InputMultiPath(Directory(), desc='Paths to add to matlabpath')
    prescript = traits.List(['ver,', 'try,'], usedefault=True, desc='prescript to be added before code')
    postscript = traits.List(['\n,catch ME,', "fprintf(2,'MATLAB code threw an exception:\\n');", "fprintf(2,'%s\\n',ME.message);", "if length(ME.stack) ~= 0, fprintf(2,'File:%s\\nName:%s\\nLine:%d\\n',ME.stack.file,ME.stack.name,ME.stack.line);, end;", 'end;'], desc='script added after code', usedefault=True)