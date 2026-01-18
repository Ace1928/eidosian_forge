from oslo_config import cfg
from oslo_log import log as logging
from glance.api import versions
from glance.common import wsgi
def _match_version_string(self, subject):
    """
        Given a string, tries to match a major and/or
        minor version number.

        :param subject: The string to check
        :returns: version found in the subject
        :raises ValueError: if no acceptable version could be found
        """
    if self.allowed_versions is None:
        self.allowed_versions = self._get_allowed_versions()
    if subject in self.allowed_versions:
        return self.allowed_versions[subject]
    else:
        raise ValueError()