from datetime import datetime
from django.conf import settings
from django.db.models.expressions import Func
from django.db.models.fields import (
from django.db.models.lookups import (
from django.utils import timezone
class Trunc(TruncBase):

    def __init__(self, expression, kind, output_field=None, tzinfo=None, **extra):
        self.kind = kind
        super().__init__(expression, output_field=output_field, tzinfo=tzinfo, **extra)