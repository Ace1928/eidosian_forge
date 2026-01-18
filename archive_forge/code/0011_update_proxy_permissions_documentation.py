import sys
from django.core.management.color import color_style
from django.db import IntegrityError, migrations, transaction
from django.db.models import Q

    Update the content_type of proxy model permissions to use the ContentType
    of the concrete model.
    