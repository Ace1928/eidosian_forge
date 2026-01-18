from copy import deepcopy as copy
from collections import namedtuple
import numpy as np
from .compat import filename_encode
from .datatype import Datatype
from .selections import SimpleSelection, select
from .. import h5d, h5p, h5s, h5t
def _get_dcpl(self, dst_filename):
    """Get the property list containing virtual dataset mappings

        If the destination filename wasn't known when the VirtualLayout was
        created, it is handled here.
        """
    dst_filename = filename_encode(dst_filename)
    if self._filename is not None:
        if dst_filename != filename_encode(self._filename):
            raise Exception(f'{dst_filename!r} != {self._filename!r}')
        return self.dcpl
    if dst_filename in self._src_filenames:
        new_dcpl = h5p.create(h5p.DATASET_CREATE)
        for i in range(self.dcpl.get_virtual_count()):
            src_filename = self.dcpl.get_virtual_filename(i)
            new_dcpl.set_virtual(self.dcpl.get_virtual_vspace(i), self._source_file_name(src_filename, dst_filename), self.dcpl.get_virtual_dsetname(i).encode('utf-8'), self.dcpl.get_virtual_srcspace(i))
        return new_dcpl
    else:
        return self.dcpl