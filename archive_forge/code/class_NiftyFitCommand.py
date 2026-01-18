import os
from ..base import CommandLine
from ...utils.filemanip import split_filename
class NiftyFitCommand(CommandLine):
    """
    Base support interface for NiftyFit commands.
    """
    _suffix = '_nf'

    def __init__(self, **inputs):
        """Init method calling super. No version to be checked."""
        super(NiftyFitCommand, self).__init__(**inputs)

    def _gen_fname(self, basename, out_dir=None, suffix=None, ext=None):
        if basename == '':
            msg = 'Unable to generate filename for command %s. ' % self.cmd
            msg += 'basename is not set!'
            raise ValueError(msg)
        _, final_bn, final_ext = split_filename(basename)
        if out_dir is None:
            out_dir = os.getcwd()
        if ext is not None:
            final_ext = ext
        if suffix is not None:
            final_bn = ''.join((final_bn, suffix))
        return os.path.abspath(os.path.join(out_dir, final_bn + final_ext))