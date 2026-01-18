import contextlib
import getpass
import logging
import os
import sqlite3
import tempfile
import warnings
from io import BytesIO
from os.path import join as pjoin
import numpy
from nibabel.optpkg import optional_package
from .nifti1 import Nifti1Header
class _Series:

    def __init__(self, d):
        self.uid = d['uid']
        self.study = d['study']
        self.number = d['number']
        self.description = d['description']
        self.rows = d['rows']
        self.columns = d['columns']
        self.bits_allocated = d['bits_allocated']
        self.bits_stored = d['bits_stored']
        self.storage_instances = None

    def __getattribute__(self, name):
        val = object.__getattribute__(self, name)
        if name == 'storage_instances' and val is None:
            val = []
            with DB.readonly_cursor() as c:
                query = 'SELECT *\n                             FROM storage_instance\n                            WHERE series = ?\n                            ORDER BY instance_number'
                c.execute(query, (self.uid,))
                cols = [el[0] for el in c.description]
                for row in c:
                    d = dict(zip(cols, row))
                    val.append(_StorageInstance(d))
            self.storage_instances = val
        return val

    def as_png(self, index=None, scale_to_slice=True):
        import PIL.Image
        if hasattr(PIL.Image, 'frombytes'):
            frombytes = PIL.Image.frombytes
        else:
            frombytes = PIL.Image.fromstring
        if index is None:
            index = len(self.storage_instances) // 2
        d = self.storage_instances[index].dicom()
        data = d.pixel_array.copy()
        if self.bits_allocated != 16:
            raise VolumeError('unsupported bits allocated')
        if self.bits_stored != 12:
            raise VolumeError('unsupported bits stored')
        data = data / 16
        if scale_to_slice:
            min = data.min()
            max = data.max()
            data = data * 255 / (max - min)
        data = data.astype(numpy.uint8)
        im = frombytes('L', (self.rows, self.columns), data.tobytes())
        s = BytesIO()
        im.save(s, 'PNG')
        return s.getvalue()

    def png_size(self, index=None, scale_to_slice=True):
        return len(self.as_png(index=index, scale_to_slice=scale_to_slice))

    def as_nifti(self):
        if len(self.storage_instances) < 2:
            raise VolumeError('too few slices')
        d = self.storage_instances[0].dicom()
        if self.bits_allocated != 16:
            raise VolumeError('unsupported bits allocated')
        if self.bits_stored != 12:
            raise VolumeError('unsupported bits stored')
        data = numpy.ndarray((len(self.storage_instances), self.rows, self.columns), dtype=numpy.int16)
        for i, si in enumerate(self.storage_instances):
            if i + 1 != si.instance_number:
                raise InstanceStackError(self, i, si)
            logger.info('reading %d/%d' % (i + 1, len(self.storage_instances)))
            d = self.storage_instances[i].dicom()
            data[i, :, :] = d.pixel_array
        d1 = self.storage_instances[0].dicom()
        dn = self.storage_instances[-1].dicom()
        pdi = d1.PixelSpacing[0]
        pdj = d1.PixelSpacing[0]
        pdk = d1.SpacingBetweenSlices
        cosi = d1.ImageOrientationPatient[0:3]
        cosi[0] = -1 * cosi[0]
        cosi[1] = -1 * cosi[1]
        cosj = d1.ImageOrientationPatient[3:6]
        cosj[0] = -1 * cosj[0]
        cosj[1] = -1 * cosj[1]
        pos_1 = numpy.array(d1.ImagePositionPatient)
        pos_1[0] = -1 * pos_1[0]
        pos_1[1] = -1 * pos_1[1]
        pos_n = numpy.array(dn.ImagePositionPatient)
        pos_n[0] = -1 * pos_n[0]
        pos_n[1] = -1 * pos_n[1]
        cosk = pos_n - pos_1
        cosk = cosk / numpy.linalg.norm(cosk)
        m = ((pdi * cosi[0], pdj * cosj[0], pdk * cosk[0], pos_1[0]), (pdi * cosi[1], pdj * cosj[1], pdk * cosk[1], pos_1[1]), (pdi * cosi[2], pdj * cosj[2], pdk * cosk[2], pos_1[2]), (0, 0, 0, 1))
        m = numpy.array(m, dtype=float)
        hdr = Nifti1Header(endianness='<')
        hdr.set_intent(0)
        hdr.set_qform(m, 1)
        hdr.set_xyzt_units(2, 8)
        hdr.set_data_dtype(numpy.int16)
        hdr.set_data_shape((self.columns, self.rows, len(self.storage_instances)))
        s = BytesIO()
        hdr.write_to(s)
        return s.getvalue() + data.tobytes()

    def nifti_size(self):
        return 352 + 2 * len(self.storage_instances) * self.columns * self.rows