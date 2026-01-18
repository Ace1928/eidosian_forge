from django.contrib.gis.db import models
from django.contrib.gis.db.backends.base.models import SpatialRefSysMixin
class OracleSpatialRefSys(models.Model, SpatialRefSysMixin):
    """Maps to the Oracle MDSYS.CS_SRS table."""
    cs_name = models.CharField(max_length=68)
    srid = models.IntegerField(primary_key=True)
    auth_srid = models.IntegerField()
    auth_name = models.CharField(max_length=256)
    wktext = models.CharField(max_length=2046)
    cs_bounds = models.PolygonField(null=True)

    class Meta:
        app_label = 'gis'
        db_table = 'CS_SRS'
        managed = False

    @property
    def wkt(self):
        return self.wktext