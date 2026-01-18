from django.apps.registry import Apps
from django.db import DatabaseError, models
from django.utils.functional import classproperty
from django.utils.timezone import now
from .exceptions import MigrationSchemaMissing
@classproperty
def Migration(cls):
    """
        Lazy load to avoid AppRegistryNotReady if installed apps import
        MigrationRecorder.
        """
    if cls._migration_class is None:

        class Migration(models.Model):
            app = models.CharField(max_length=255)
            name = models.CharField(max_length=255)
            applied = models.DateTimeField(default=now)

            class Meta:
                apps = Apps()
                app_label = 'migrations'
                db_table = 'django_migrations'

            def __str__(self):
                return 'Migration %s for %s' % (self.name, self.app)
        cls._migration_class = Migration
    return cls._migration_class