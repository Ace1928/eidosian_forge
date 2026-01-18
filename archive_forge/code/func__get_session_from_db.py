import logging
from django.contrib.sessions.backends.base import CreateError, SessionBase, UpdateError
from django.core.exceptions import SuspiciousOperation
from django.db import DatabaseError, IntegrityError, router, transaction
from django.utils import timezone
from django.utils.functional import cached_property
def _get_session_from_db(self):
    try:
        return self.model.objects.get(session_key=self.session_key, expire_date__gt=timezone.now())
    except (self.model.DoesNotExist, SuspiciousOperation) as e:
        if isinstance(e, SuspiciousOperation):
            logger = logging.getLogger('django.security.%s' % e.__class__.__name__)
            logger.warning(str(e))
        self._session_key = None