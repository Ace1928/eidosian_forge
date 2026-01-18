from django.contrib.gis.gdal import DataSource
from django.contrib.gis.gdal.field import (
def _ogrinspect(data_source, model_name, geom_name='geom', layer_key=0, srid=None, multi_geom=False, name_field=None, imports=True, decimal=False, blank=False, null=False):
    """
    Helper routine for `ogrinspect` that generates GeoDjango models corresponding
    to the given data source.  See the `ogrinspect` docstring for more details.
    """
    if isinstance(data_source, str):
        data_source = DataSource(data_source)
    elif isinstance(data_source, DataSource):
        pass
    else:
        raise TypeError('Data source parameter must be a string or a DataSource object.')
    layer = data_source[layer_key]
    ogr_fields = layer.fields

    def process_kwarg(kwarg):
        if isinstance(kwarg, (list, tuple)):
            return [s.lower() for s in kwarg]
        elif kwarg:
            return [s.lower() for s in ogr_fields]
        else:
            return []
    null_fields = process_kwarg(null)
    blank_fields = process_kwarg(blank)
    decimal_fields = process_kwarg(decimal)

    def get_kwargs_str(field_name):
        kwlist = []
        if field_name.lower() in null_fields:
            kwlist.append('null=True')
        if field_name.lower() in blank_fields:
            kwlist.append('blank=True')
        if kwlist:
            return ', ' + ', '.join(kwlist)
        else:
            return ''
    if imports:
        yield '# This is an auto-generated Django model module created by ogrinspect.'
        yield 'from django.contrib.gis.db import models'
        yield ''
        yield ''
    yield ('class %s(models.Model):' % model_name)
    for field_name, width, precision, field_type in zip(ogr_fields, layer.field_widths, layer.field_precisions, layer.field_types):
        mfield = field_name.lower()
        if mfield[-1:] == '_':
            mfield += 'field'
        kwargs_str = get_kwargs_str(field_name)
        if field_type is OFTReal:
            if field_name.lower() in decimal_fields:
                yield ('    %s = models.DecimalField(max_digits=%d, decimal_places=%d%s)' % (mfield, width, precision, kwargs_str))
            else:
                yield ('    %s = models.FloatField(%s)' % (mfield, kwargs_str[2:]))
        elif field_type is OFTInteger:
            yield ('    %s = models.IntegerField(%s)' % (mfield, kwargs_str[2:]))
        elif field_type is OFTInteger64:
            yield ('    %s = models.BigIntegerField(%s)' % (mfield, kwargs_str[2:]))
        elif field_type is OFTString:
            yield ('    %s = models.CharField(max_length=%s%s)' % (mfield, width, kwargs_str))
        elif field_type is OFTDate:
            yield ('    %s = models.DateField(%s)' % (mfield, kwargs_str[2:]))
        elif field_type is OFTDateTime:
            yield ('    %s = models.DateTimeField(%s)' % (mfield, kwargs_str[2:]))
        elif field_type is OFTTime:
            yield ('    %s = models.TimeField(%s)' % (mfield, kwargs_str[2:]))
        else:
            raise TypeError('Unknown field type %s in %s' % (field_type, mfield))
    gtype = layer.geom_type
    if multi_geom:
        gtype.to_multi()
    geom_field = gtype.django
    if srid is None:
        if layer.srs is None:
            srid_str = 'srid=-1'
        else:
            srid = layer.srs.srid
            if srid is None:
                srid_str = 'srid=-1'
            elif srid == 4326:
                srid_str = ''
            else:
                srid_str = 'srid=%s' % srid
    else:
        srid_str = 'srid=%s' % srid
    yield ('    %s = models.%s(%s)' % (geom_name, geom_field, srid_str))
    if name_field:
        yield ''
        yield ('    def __str__(self): return self.%s' % name_field)