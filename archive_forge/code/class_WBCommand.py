import os
import re
from ... import logging
from ...utils.filemanip import split_filename
from ..base import CommandLine, PackageInfo
class WBCommand(CommandLine):
    """Base support for workbench commands."""

    @property
    def version(self):
        return Info.version()

    def _gen_filename(self, name, outdir=None, suffix='', ext=None):
        """Generate a filename based on the given parameters.
        The filename will take the form: <basename><suffix><ext>.
        Parameters
        ----------
        name : str
            Filename to base the new filename on.
        suffix : str
            Suffix to add to the `basename`.  (defaults is '' )
        ext : str
            Extension to use for the new filename.
        Returns
        -------
        fname : str
            New filename based on given parameters.
        """
        if not name:
            raise ValueError('Cannot generate filename - filename not set')
        _, fname, fext = split_filename(name)
        if ext is None:
            ext = fext
        if outdir is None:
            outdir = '.'
        return os.path.join(outdir, fname + suffix + ext)