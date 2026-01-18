from django.apps import apps
from django.contrib.gis.db.models import GeometryField
from django.contrib.sitemaps import Sitemap
from django.db import models
from django.urls import reverse
def _build_kml_sources(self, sources):
    """
        Go through the given sources and return a 3-tuple of the application
        label, module name, and field name of every GeometryField encountered
        in the sources.

        If no sources are provided, then all models.
        """
    kml_sources = []
    if sources is None:
        sources = apps.get_models()
    for source in sources:
        if isinstance(source, models.base.ModelBase):
            for field in source._meta.fields:
                if isinstance(field, GeometryField):
                    kml_sources.append((source._meta.app_label, source._meta.model_name, field.name))
        elif isinstance(source, (list, tuple)):
            if len(source) != 3:
                raise ValueError('Must specify a 3-tuple of (app_label, module_name, field_name).')
            kml_sources.append(source)
        else:
            raise TypeError('KML Sources must be a model or a 3-tuple.')
    return kml_sources