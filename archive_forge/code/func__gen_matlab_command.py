import os
from .. import config
from .base import (
def _gen_matlab_command(self, argstr, script_lines):
    """Generates commands and, if mfile specified, writes it to disk."""
    cwd = os.getcwd()
    mfile = self.inputs.mfile or self.inputs.uses_mcr
    paths = []
    if isdefined(self.inputs.paths):
        paths = self.inputs.paths
    prescript = self.inputs.prescript
    postscript = self.inputs.postscript
    if mfile:
        prescript.insert(0, "fprintf(1,'Executing %s at %s:\\n',mfilename(),datestr(now));")
    else:
        prescript.insert(0, "fprintf(1,'Executing code at %s:\\n',datestr(now));")
    for path in paths:
        prescript.append("if ~(ismcc || isdeployed), addpath('%s'); end;\n" % path)
    if not mfile:
        script_lines = ','.join([line for line in script_lines.split('\n') if not line.strip().startswith('%')])
    script_lines = '\n'.join(prescript) + script_lines + '\n'.join(postscript)
    if mfile:
        with open(os.path.join(cwd, self.inputs.script_file), 'wt') as mfile:
            mfile.write(script_lines)
        if self.inputs.uses_mcr:
            script = '%s' % os.path.join(cwd, self.inputs.script_file)
        else:
            script = "addpath('%s');%s" % (cwd, self.inputs.script_file.split('.')[0])
    else:
        script = ''.join(script_lines.split('\n'))
    return argstr % script