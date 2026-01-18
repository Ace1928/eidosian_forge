import numpy as np
import tempfile
from ...utils.misc import package_check
from ...utils.filemanip import fname_presuffix
from .base import NitimeBaseInterface
from ..base import (
def _make_output_files(self):
    """
        Generate the output csv files.
        """
    for this in zip([self.coherence, self.delay], ['coherence', 'delay']):
        tmp_f = tempfile.mkstemp()[1]
        np.savetxt(tmp_f, this[0], delimiter=',')
        fid = open(fname_presuffix(self.inputs.output_csv_file, suffix='_%s' % this[1]), 'w+')
        fid.write(',' + ','.join(self.ROIs) + '\n')
        for r, line in zip(self.ROIs, open(tmp_f)):
            fid.write('%s,%s' % (r, line))
        fid.close()