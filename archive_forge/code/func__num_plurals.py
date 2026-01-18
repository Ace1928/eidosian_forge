import json
import os
import re
from pathlib import Path
from django.apps import apps
from django.conf import settings
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.template import Context, Engine
from django.urls import translate_url
from django.utils.formats import get_format
from django.utils.http import url_has_allowed_host_and_scheme
from django.utils.translation import check_for_language, get_language
from django.utils.translation.trans_real import DjangoTranslation
from django.views.generic import View
@property
def _num_plurals(self):
    """
        Return the number of plurals for this catalog language, or 2 if no
        plural string is available.
        """
    match = re.search('nplurals=\\s*(\\d+)', self._plural_string or '')
    if match:
        return int(match[1])
    return 2