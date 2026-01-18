from collections import defaultdict
from django.apps import apps
from django.db import models
from django.db.models import Q
from django.utils.translation import gettext_lazy as _
def _get_from_cache(self, opts):
    key = (opts.app_label, opts.model_name)
    return self._cache[self.db][key]