import os
from ...utils.filemanip import split_filename
from ..base import CommandLineInputSpec, CommandLine, traits, TraitedSpec, File
class Trackvis2CaminoInputSpec(CommandLineInputSpec):
    """Wraps trackvis_to_camino from Camino-Trackvis

    Convert files from camino .Bfloat format to trackvis .trk format.

    Example
    -------

    >>> import nipype.interfaces.camino2trackvis as cam2trk
    >>> t2c = cam2trk.Trackvis2Camino()
    >>> t2c.inputs.in_file = 'streamlines.trk'
    >>> t2c.inputs.out_file = 'streamlines.Bfloat'
    >>> t2c.run()                  # doctest: +SKIP
    """
    in_file = File(exists=True, argstr='-i %s', mandatory=True, position=1, desc='The input .trk (trackvis) file.')
    out_file = File(argstr='-o %s', genfile=True, position=2, desc='The filename to which to write the .Bfloat (camino).')
    append_file = File(exists=True, argstr='-a %s', position=2, desc='A file to which the append the .Bfloat data. ')