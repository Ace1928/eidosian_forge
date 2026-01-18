from django.contrib.gis.db.backends.base.models import SpatialRefSysMixin
from django.db import models
class PostGISSpatialRefSys(models.Model, SpatialRefSysMixin):
    """
    The 'spatial_ref_sys' table from PostGIS. See the PostGIS
    documentation at Ch. 4.2.1.
    """
    srid = models.IntegerField(primary_key=True)
    auth_name = models.CharField(max_length=256)
    auth_srid = models.IntegerField()
    srtext = models.CharField(max_length=2048)
    proj4text = models.CharField(max_length=2048)

    class Meta:
        app_label = 'gis'
        db_table = 'spatial_ref_sys'
        managed = False

    @property
    def wkt(self):
        return self.srtext