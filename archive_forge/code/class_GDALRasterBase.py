from django.contrib.gis.gdal.base import GDALBase
from django.contrib.gis.gdal.prototypes import raster as capi
class GDALRasterBase(GDALBase):
    """
    Attributes that exist on both GDALRaster and GDALBand.
    """

    @property
    def metadata(self):
        """
        Return the metadata for this raster or band. The return value is a
        nested dictionary, where the first-level key is the metadata domain and
        the second-level is the metadata item names and values for that domain.
        """
        domain_list = ['DEFAULT']
        meta_list = capi.get_ds_metadata_domain_list(self._ptr)
        if meta_list:
            counter = 0
            domain = meta_list[counter]
            while domain:
                domain_list.append(domain.decode())
                counter += 1
                domain = meta_list[counter]
        capi.free_dsl(meta_list)
        result = {}
        for domain in domain_list:
            data = capi.get_ds_metadata(self._ptr, None if domain == 'DEFAULT' else domain.encode())
            if not data:
                continue
            domain_meta = {}
            counter = 0
            item = data[counter]
            while item:
                key, val = item.decode().split('=')
                domain_meta[key] = val
                counter += 1
                item = data[counter]
            result[domain or 'DEFAULT'] = domain_meta
        return result

    @metadata.setter
    def metadata(self, value):
        """
        Set the metadata. Update only the domains that are contained in the
        value dictionary.
        """
        for domain, metadata in value.items():
            domain = None if domain == 'DEFAULT' else domain.encode()
            for meta_name, meta_value in metadata.items():
                capi.set_ds_metadata_item(self._ptr, meta_name.encode(), meta_value.encode() if meta_value else None, domain)