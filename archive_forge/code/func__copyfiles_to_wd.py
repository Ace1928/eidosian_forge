from collections import OrderedDict, defaultdict
import os
import os.path as op
from pathlib import Path
import shutil
import socket
from copy import deepcopy
from glob import glob
from logging import INFO
from tempfile import mkdtemp
from ... import config, logging
from ...utils.misc import flatten, unflatten, str2bool, dict_diff
from ...utils.filemanip import (
from ...interfaces.base import (
from ...interfaces.base.specs import get_filecopy_info
from .utils import (
from .base import EngineBase
def _copyfiles_to_wd(self, execute=True, linksonly=False):
    """copy files over and change the inputs"""
    filecopy_info = get_filecopy_info(self.interface)
    if not filecopy_info:
        return
    logger.debug('copying files to wd [execute=%s, linksonly=%s]', execute, linksonly)
    outdir = self.output_dir()
    if execute and linksonly:
        olddir = outdir
        outdir = op.join(outdir, '_tempinput')
        os.makedirs(outdir, exist_ok=True)
    for info in filecopy_info:
        files = self.inputs.trait_get().get(info['key'])
        if not isdefined(files) or not files:
            continue
        infiles = ensure_list(files)
        if execute:
            if linksonly:
                if not info['copy']:
                    newfiles = copyfiles(infiles, [outdir], copy=info['copy'], create_new=True)
                else:
                    newfiles = fnames_presuffix(infiles, newpath=outdir)
                newfiles = _strip_temp(newfiles, op.abspath(olddir).split(op.sep)[-1])
            else:
                newfiles = copyfiles(infiles, [outdir], copy=info['copy'], create_new=True)
        else:
            newfiles = fnames_presuffix(infiles, newpath=outdir)
        if not isinstance(files, list):
            newfiles = simplify_list(newfiles)
        setattr(self.inputs, info['key'], newfiles)
    if execute and linksonly:
        emptydirs(outdir, noexist_ok=True)