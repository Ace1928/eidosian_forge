import os
import glob
from ...utils.filemanip import split_filename
from ..base import (
class Shredder(StdOutCommandLine):
    """
    Extracts periodic chunks from a data stream.

    Shredder makes an initial offset of offset bytes. It then reads and outputs
    chunksize bytes, skips space bytes, and repeats until there is no more input.

    If  the  chunksize  is  negative, chunks of size chunksize are read and the
    byte ordering of each chunk is reversed. The whole chunk will be reversed, so
    the chunk must be the same size as the data type, otherwise the order of the
    values in the chunk, as well as their endianness, will be reversed.

    Examples
    --------

    >>> import nipype.interfaces.camino as cam
    >>> shred = cam.Shredder()
    >>> shred.inputs.in_file = 'SubjectA.Bfloat'
    >>> shred.inputs.offset = 0
    >>> shred.inputs.chunksize = 1
    >>> shred.inputs.space = 2
    >>> shred.run()                  # doctest: +SKIP
    """
    _cmd = 'shredder'
    input_spec = ShredderInputSpec
    output_spec = ShredderOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['shredded_file'] = os.path.abspath(self._gen_outfilename())
        return outputs

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.in_file)
        return name + '_shredded'