from django import forms
from django.conf import settings
from django.contrib.flatpages.models import FlatPage
from django.core.exceptions import ValidationError
from django.utils.translation import gettext
from django.utils.translation import gettext_lazy as _
def _trailing_slash_required(self):
    return settings.APPEND_SLASH and 'django.middleware.common.CommonMiddleware' in settings.MIDDLEWARE