import urllib.parse as urlparse
from glance.i18n import _
class InvalidDataMigrationScript(GlanceException):
    message = _("Invalid data migration script '%(script)s'. A valid data migration script must implement functions 'has_migrations' and 'migrate'.")