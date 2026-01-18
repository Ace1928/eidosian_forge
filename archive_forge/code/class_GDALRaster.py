import json
import os
import sys
import uuid
from ctypes import (
from pathlib import Path
from django.contrib.gis.gdal.driver import Driver
from django.contrib.gis.gdal.error import GDALException
from django.contrib.gis.gdal.prototypes import raster as capi
from django.contrib.gis.gdal.raster.band import BandList
from django.contrib.gis.gdal.raster.base import GDALRasterBase
from django.contrib.gis.gdal.raster.const import (
from django.contrib.gis.gdal.srs import SpatialReference, SRSException
from django.contrib.gis.geometry import json_regex
from django.utils.encoding import force_bytes, force_str
from django.utils.functional import cached_property
class GDALRaster(GDALRasterBase):
    """
    Wrap a raster GDAL Data Source object.
    """
    destructor = capi.close_ds

    def __init__(self, ds_input, write=False):
        self._write = 1 if write else 0
        Driver.ensure_registered()
        if isinstance(ds_input, str) and json_regex.match(ds_input):
            ds_input = json.loads(ds_input)
        if isinstance(ds_input, (str, Path)):
            ds_input = str(ds_input)
            if not ds_input.startswith(VSI_FILESYSTEM_PREFIX) and (not os.path.exists(ds_input)):
                raise GDALException('Unable to read raster source input "%s".' % ds_input)
            try:
                self._ptr = capi.open_ds(force_bytes(ds_input), self._write)
            except GDALException as err:
                raise GDALException('Could not open the datasource at "{}" ({}).'.format(ds_input, err))
        elif isinstance(ds_input, bytes):
            self._write = 1
            size = sys.getsizeof(ds_input)
            self._ds_input = c_buffer(ds_input)
            vsi_path = os.path.join(VSI_MEM_FILESYSTEM_BASE_PATH, str(uuid.uuid4()))
            capi.create_vsi_file_from_mem_buffer(force_bytes(vsi_path), byref(self._ds_input), size, VSI_TAKE_BUFFER_OWNERSHIP)
            try:
                self._ptr = capi.open_ds(force_bytes(vsi_path), self._write)
            except GDALException:
                capi.unlink_vsi_file(force_bytes(vsi_path))
                raise GDALException('Failed creating VSI raster from the input buffer.')
        elif isinstance(ds_input, dict):
            self._write = 1
            driver = Driver(ds_input.get('driver', 'MEM'))
            if driver.name != 'MEM' and 'name' not in ds_input:
                raise GDALException('Specify name for creation of raster with driver "{}".'.format(driver.name))
            if 'width' not in ds_input or 'height' not in ds_input:
                raise GDALException('Specify width and height attributes for JSON or dict input.')
            if 'srid' not in ds_input:
                raise GDALException('Specify srid for JSON or dict input.')
            papsz_options = []
            for key, val in ds_input.get('papsz_options', {}).items():
                option = '{}={}'.format(key, val)
                papsz_options.append(option.upper().encode())
            papsz_options.append(None)
            papsz_options = (c_char_p * len(papsz_options))(*papsz_options)
            self._ptr = capi.create_ds(driver._ptr, force_bytes(ds_input.get('name', '')), ds_input['width'], ds_input['height'], ds_input.get('nr_of_bands', len(ds_input.get('bands', []))), ds_input.get('datatype', 6), byref(papsz_options))
            for i, band_input in enumerate(ds_input.get('bands', [])):
                band = self.bands[i]
                if 'nodata_value' in band_input:
                    band.nodata_value = band_input['nodata_value']
                    if band.nodata_value is not None and ('data' not in band_input or 'size' in band_input or 'shape' in band_input):
                        band.data(data=(band.nodata_value,), shape=(1, 1))
                band.data(data=band_input.get('data'), size=band_input.get('size'), shape=band_input.get('shape'), offset=band_input.get('offset'))
            self.srs = ds_input.get('srid')
            if 'origin' in ds_input:
                self.origin.x, self.origin.y = ds_input['origin']
            if 'scale' in ds_input:
                self.scale.x, self.scale.y = ds_input['scale']
            if 'skew' in ds_input:
                self.skew.x, self.skew.y = ds_input['skew']
        elif isinstance(ds_input, c_void_p):
            self._ptr = ds_input
        else:
            raise GDALException('Invalid data source input type: "{}".'.format(type(ds_input)))

    def __del__(self):
        if self.is_vsi_based:
            capi.unlink_vsi_file(force_bytes(self.name))
        super().__del__()

    def __str__(self):
        return self.name

    def __repr__(self):
        """
        Short-hand representation because WKB may be very large.
        """
        return '<Raster object at %s>' % hex(addressof(self._ptr))

    def _flush(self):
        """
        Flush all data from memory into the source file if it exists.
        The data that needs flushing are geotransforms, coordinate systems,
        nodata_values and pixel values. This function will be called
        automatically wherever it is needed.
        """
        if not self._write:
            raise GDALException('Raster needs to be opened in write mode to change values.')
        capi.flush_ds(self._ptr)

    @property
    def vsi_buffer(self):
        if not (self.is_vsi_based and self.name.startswith(VSI_MEM_FILESYSTEM_BASE_PATH)):
            return None
        out_length = c_int()
        dat = capi.get_mem_buffer_from_vsi_file(force_bytes(self.name), byref(out_length), VSI_DELETE_BUFFER_ON_READ)
        return string_at(dat, out_length.value)

    @cached_property
    def is_vsi_based(self):
        return self._ptr and self.name.startswith(VSI_FILESYSTEM_PREFIX)

    @property
    def name(self):
        """
        Return the name of this raster. Corresponds to filename
        for file-based rasters.
        """
        return force_str(capi.get_ds_description(self._ptr))

    @cached_property
    def driver(self):
        """
        Return the GDAL Driver used for this raster.
        """
        ds_driver = capi.get_ds_driver(self._ptr)
        return Driver(ds_driver)

    @property
    def width(self):
        """
        Width (X axis) in pixels.
        """
        return capi.get_ds_xsize(self._ptr)

    @property
    def height(self):
        """
        Height (Y axis) in pixels.
        """
        return capi.get_ds_ysize(self._ptr)

    @property
    def srs(self):
        """
        Return the SpatialReference used in this GDALRaster.
        """
        try:
            wkt = capi.get_ds_projection_ref(self._ptr)
            if not wkt:
                return None
            return SpatialReference(wkt, srs_type='wkt')
        except SRSException:
            return None

    @srs.setter
    def srs(self, value):
        """
        Set the spatial reference used in this GDALRaster. The input can be
        a SpatialReference or any parameter accepted by the SpatialReference
        constructor.
        """
        if isinstance(value, SpatialReference):
            srs = value
        elif isinstance(value, (int, str)):
            srs = SpatialReference(value)
        else:
            raise ValueError('Could not create a SpatialReference from input.')
        capi.set_ds_projection_ref(self._ptr, srs.wkt.encode())
        self._flush()

    @property
    def srid(self):
        """
        Shortcut to access the srid of this GDALRaster.
        """
        return self.srs.srid

    @srid.setter
    def srid(self, value):
        """
        Shortcut to set this GDALRaster's srs from an srid.
        """
        self.srs = value

    @property
    def geotransform(self):
        """
        Return the geotransform of the data source.
        Return the default geotransform if it does not exist or has not been
        set previously. The default is [0.0, 1.0, 0.0, 0.0, 0.0, -1.0].
        """
        gtf = (c_double * 6)()
        capi.get_ds_geotransform(self._ptr, byref(gtf))
        return list(gtf)

    @geotransform.setter
    def geotransform(self, values):
        """Set the geotransform for the data source."""
        if len(values) != 6 or not all((isinstance(x, (int, float)) for x in values)):
            raise ValueError('Geotransform must consist of 6 numeric values.')
        values = (c_double * 6)(*values)
        capi.set_ds_geotransform(self._ptr, byref(values))
        self._flush()

    @property
    def origin(self):
        """
        Coordinates of the raster origin.
        """
        return TransformPoint(self, 'origin')

    @property
    def scale(self):
        """
        Pixel scale in units of the raster projection.
        """
        return TransformPoint(self, 'scale')

    @property
    def skew(self):
        """
        Skew of pixels (rotation parameters).
        """
        return TransformPoint(self, 'skew')

    @property
    def extent(self):
        """
        Return the extent as a 4-tuple (xmin, ymin, xmax, ymax).
        """
        xval = self.origin.x + self.scale.x * self.width
        yval = self.origin.y + self.scale.y * self.height
        xmin = min(xval, self.origin.x)
        xmax = max(xval, self.origin.x)
        ymin = min(yval, self.origin.y)
        ymax = max(yval, self.origin.y)
        return (xmin, ymin, xmax, ymax)

    @property
    def bands(self):
        return BandList(self)

    def warp(self, ds_input, resampling='NearestNeighbour', max_error=0.0):
        """
        Return a warped GDALRaster with the given input characteristics.

        The input is expected to be a dictionary containing the parameters
        of the target raster. Allowed values are width, height, SRID, origin,
        scale, skew, datatype, driver, and name (filename).

        By default, the warp functions keeps all parameters equal to the values
        of the original source raster. For the name of the target raster, the
        name of the source raster will be used and appended with
        _copy. + source_driver_name.

        In addition, the resampling algorithm can be specified with the "resampling"
        input parameter. The default is NearestNeighbor. For a list of all options
        consult the GDAL_RESAMPLE_ALGORITHMS constant.
        """
        ds_input.setdefault('width', self.width)
        ds_input.setdefault('height', self.height)
        ds_input.setdefault('srid', self.srs.srid)
        ds_input.setdefault('origin', self.origin)
        ds_input.setdefault('scale', self.scale)
        ds_input.setdefault('skew', self.skew)
        ds_input.setdefault('driver', self.driver.name)
        if 'name' not in ds_input:
            ds_input['name'] = self.name + '_copy.' + self.driver.name
        if 'datatype' not in ds_input:
            ds_input['datatype'] = self.bands[0].datatype()
        ds_input['bands'] = [{'nodata_value': bnd.nodata_value} for bnd in self.bands]
        target = GDALRaster(ds_input, write=True)
        algorithm = GDAL_RESAMPLE_ALGORITHMS[resampling]
        capi.reproject_image(self._ptr, self.srs.wkt.encode(), target._ptr, target.srs.wkt.encode(), algorithm, 0.0, max_error, c_void_p(), c_void_p(), c_void_p())
        target._flush()
        return target

    def clone(self, name=None):
        """Return a clone of this GDALRaster."""
        if name:
            clone_name = name
        elif self.driver.name != 'MEM':
            clone_name = self.name + '_copy.' + self.driver.name
        else:
            clone_name = os.path.join(VSI_MEM_FILESYSTEM_BASE_PATH, str(uuid.uuid4()))
        return GDALRaster(capi.copy_ds(self.driver._ptr, force_bytes(clone_name), self._ptr, c_int(), c_char_p(), c_void_p(), c_void_p()), write=self._write)

    def transform(self, srs, driver=None, name=None, resampling='NearestNeighbour', max_error=0.0):
        """
        Return a copy of this raster reprojected into the given spatial
        reference system.
        """
        algorithm = GDAL_RESAMPLE_ALGORITHMS[resampling]
        if isinstance(srs, SpatialReference):
            target_srs = srs
        elif isinstance(srs, (int, str)):
            target_srs = SpatialReference(srs)
        else:
            raise TypeError('Transform only accepts SpatialReference, string, and integer objects.')
        if target_srs.srid == self.srid and (not driver or driver == self.driver.name):
            return self.clone(name)
        target = capi.auto_create_warped_vrt(self._ptr, self.srs.wkt.encode(), target_srs.wkt.encode(), algorithm, max_error, c_void_p())
        target = GDALRaster(target)
        data = {'srid': target_srs.srid, 'width': target.width, 'height': target.height, 'origin': [target.origin.x, target.origin.y], 'scale': [target.scale.x, target.scale.y], 'skew': [target.skew.x, target.skew.y]}
        if driver:
            data['driver'] = driver
        if name:
            data['name'] = name
        return self.warp(data, resampling=resampling, max_error=max_error)

    @property
    def info(self):
        """
        Return information about this raster in a string format equivalent
        to the output of the gdalinfo command line utility.
        """
        return capi.get_ds_info(self.ptr, None).decode()