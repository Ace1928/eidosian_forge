from django.db import models
from django.utils.translation import gettext_lazy as _
class AbstractBaseSession(models.Model):
    session_key = models.CharField(_('session key'), max_length=40, primary_key=True)
    session_data = models.TextField(_('session data'))
    expire_date = models.DateTimeField(_('expire date'), db_index=True)
    objects = BaseSessionManager()

    class Meta:
        abstract = True
        verbose_name = _('session')
        verbose_name_plural = _('sessions')

    def __str__(self):
        return self.session_key

    @classmethod
    def get_session_store_class(cls):
        raise NotImplementedError

    def get_decoded(self):
        session_store_class = self.get_session_store_class()
        return session_store_class().decode(self.session_data)