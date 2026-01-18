import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class MigrationNotProvided(Exception):

    def __init__(self, mod_name, path):
        super(MigrationNotProvided, self).__init__(_("%(mod_name)s doesn't provide database migrations. The migration repository path at %(path)s doesn't exist or isn't a directory.") % {'mod_name': mod_name, 'path': path})