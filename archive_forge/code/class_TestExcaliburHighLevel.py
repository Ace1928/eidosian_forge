import numpy as np
from numpy.testing import assert_array_equal
import os
import os.path as osp
import shutil
import tempfile
import h5py as h5
from ..common import ut
from ..._hl.vds import vds_support
@ut.skipUnless(vds_support, 'VDS requires HDF5 >= 1.9.233')
class TestExcaliburHighLevel(ut.TestCase):

    def create_excalibur_fem_stripe_datafile(self, fname, nframes, excalibur_data, scale):
        shape = (nframes,) + excalibur_data.fem_stripe_dimensions
        max_shape = shape
        chunk = (1,) + excalibur_data.fem_stripe_dimensions
        with h5.File(fname, 'w', libver='latest') as f:
            dset = f.create_dataset('data', shape=shape, maxshape=max_shape, chunks=chunk, dtype='uint16')
            for data_value_index in np.arange(nframes):
                dset[data_value_index] = excalibur_data.generate_fem_stripe_image(data_value_index * scale)

    def setUp(self):
        self.working_dir = tempfile.mkdtemp()
        self.fname = ['stripe_%d.h5' % stripe for stripe in range(1, 7)]
        self.fname = [osp.join(self.working_dir, f) for f in self.fname]
        nframes = 5
        self.edata = ExcaliburData()
        for k, raw_file in enumerate(self.fname):
            self.create_excalibur_fem_stripe_datafile(raw_file, nframes, self.edata, k)

    def test_excalibur_high_level(self):
        outfile = osp.join(self.working_dir, 'excalibur.h5')
        f = h5.File(outfile, 'w', libver='latest')
        in_key = 'data'
        in_sh = h5.File(self.fname[0], 'r')[in_key].shape
        dtype = h5.File(self.fname[0], 'r')[in_key].dtype
        vertical_gap = 10
        nfiles = len(self.fname)
        nframes = in_sh[0]
        width = in_sh[2]
        height = in_sh[1] * nfiles + vertical_gap * (nfiles - 1)
        out_sh = (nframes, height, width)
        layout = h5.VirtualLayout(shape=out_sh, dtype=dtype)
        offset = 0
        for i, filename in enumerate(self.fname):
            vsource = h5.VirtualSource(filename, in_key, shape=in_sh)
            layout[:, offset:offset + in_sh[1], :] = vsource
            offset += in_sh[1] + vertical_gap
        f.create_virtual_dataset('data', layout, fillvalue=1)
        f.close()
        f = h5.File(outfile, 'r')['data']
        self.assertEqual(f[3, 100, 0], 0.0)
        self.assertEqual(f[3, 260, 0], 1.0)
        self.assertEqual(f[3, 350, 0], 3.0)
        self.assertEqual(f[3, 650, 0], 6.0)
        self.assertEqual(f[3, 900, 0], 9.0)
        self.assertEqual(f[3, 1150, 0], 12.0)
        self.assertEqual(f[3, 1450, 0], 15.0)
        f.file.close()

    def tearDown(self):
        shutil.rmtree(self.working_dir)