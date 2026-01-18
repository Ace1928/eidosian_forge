import os
from ...utils.filemanip import split_filename
from ..base import (
class Track(CommandLine):
    """
    Performs tractography using one of the following models:
    dt', 'multitensor', 'pds', 'pico', 'bootstrap', 'ballstick', 'bayesdirac'

    Example
    -------
    >>> import nipype.interfaces.camino as cmon
    >>> track = cmon.Track()
    >>> track.inputs.inputmodel = 'dt'
    >>> track.inputs.in_file = 'data.Bfloat'
    >>> track.inputs.seed_file = 'seed_mask.nii'
    >>> track.run()                  # doctest: +SKIP

    """
    _cmd = 'track'
    input_spec = TrackInputSpec
    output_spec = TrackOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.inputs.out_file):
            out_file_path = os.path.abspath(self.inputs.out_file)
        else:
            out_file_path = os.path.abspath(self._gen_outfilename())
        outputs['tracked'] = out_file_path
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._gen_outfilename()
        else:
            return None

    def _gen_outfilename(self):
        if not isdefined(self.inputs.in_file):
            name = 'bedpostx'
        else:
            _, name, _ = split_filename(self.inputs.in_file)
        return name + '_tracked'