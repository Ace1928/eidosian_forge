import os
from ... import logging
from ...utils.filemanip import fname_presuffix
from ..base import CommandLine
from nipype.interfaces.fsl.base import Info
import warnings
class CommandLineDtitk(CommandLine):

    def _gen_fname(self, basename, cwd=None, suffix=None, change_ext=True, ext=None):
        """Generate a filename based on the given parameters.

        The filename will take the form: cwd/basename<suffix><ext>.
        If change_ext is True, it will use the extensions specified in
        <instance>inputs.output_type.

        Parameters
        ----------
        basename : str
            Filename to base the new filename on.
        cwd : str
            Path to prefix to the new filename. (default is os.getcwd())
        suffix : str
            Suffix to add to the `basename`.  (defaults is '' )
        change_ext : bool
            Flag to change the filename extension to the FSL output type.
            (default True)

        Returns
        -------
        fname : str
            New filename based on given parameters.

        """
        if basename == '':
            msg = 'Unable to generate filename for command %s. ' % self.cmd
            msg += 'basename is not set!'
            raise ValueError(msg)
        if cwd is None:
            cwd = os.getcwd()
        if ext is None:
            ext = Info.output_type_to_ext(self.inputs.output_type)
        if change_ext:
            if suffix:
                suffix = ''.join((suffix, ext))
            else:
                suffix = ext
        if suffix is None:
            suffix = ''
        fname = fname_presuffix(basename, suffix=suffix, use_ext=False, newpath=cwd)
        return fname