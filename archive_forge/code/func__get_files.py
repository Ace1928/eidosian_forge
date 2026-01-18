import os.path as op
import glob
from ... import logging
from ...utils.filemanip import simplify_list
from ..base import traits, File, Directory, TraitedSpec, OutputMultiPath
from ..freesurfer.base import FSCommand, FSTraitedSpec
def _get_files(self, path, key, dirval, altkey=None):
    globsuffix = '*'
    globprefix = '*'
    keydir = op.join(path, dirval)
    if altkey:
        key = altkey
    globpattern = op.join(keydir, ''.join((globprefix, key, globsuffix)))
    return glob.glob(globpattern)