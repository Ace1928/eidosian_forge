from django.apps import apps as django_apps
from django.conf import settings
from django.core import paginator
from django.core.exceptions import ImproperlyConfigured
from django.utils import translation
class GenericSitemap(Sitemap):
    priority = None
    changefreq = None

    def __init__(self, info_dict, priority=None, changefreq=None, protocol=None):
        self.queryset = info_dict['queryset']
        self.date_field = info_dict.get('date_field')
        self.priority = self.priority or priority
        self.changefreq = self.changefreq or changefreq
        self.protocol = self.protocol or protocol

    def items(self):
        return self.queryset.filter()

    def lastmod(self, item):
        if self.date_field is not None:
            return getattr(item, self.date_field)
        return None

    def get_latest_lastmod(self):
        if self.date_field is not None:
            return self.queryset.order_by('-' + self.date_field).values_list(self.date_field, flat=True).first()
        return None