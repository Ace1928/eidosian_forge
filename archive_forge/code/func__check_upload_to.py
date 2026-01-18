import datetime
import posixpath
from django import forms
from django.core import checks
from django.core.files.base import File
from django.core.files.images import ImageFile
from django.core.files.storage import Storage, default_storage
from django.core.files.utils import validate_file_name
from django.db.models import signals
from django.db.models.fields import Field
from django.db.models.query_utils import DeferredAttribute
from django.db.models.utils import AltersData
from django.utils.translation import gettext_lazy as _
def _check_upload_to(self):
    if isinstance(self.upload_to, str) and self.upload_to.startswith('/'):
        return [checks.Error("%s's 'upload_to' argument must be a relative path, not an absolute path." % self.__class__.__name__, obj=self, id='fields.E202', hint='Remove the leading slash.')]
    else:
        return []