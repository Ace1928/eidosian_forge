import sys
import os
import re
from email import message_from_file
from distutils.errors import *
from distutils.fancy_getopt import FancyGetopt, translate_longopt
from distutils.util import check_environ, strtobool, rfc822_escape
from distutils import log
from distutils.debug import DEBUG
class DistributionMetadata:
    """Dummy class to hold the distribution meta-data: name, version,
    author, and so forth.
    """
    _METHOD_BASENAMES = ('name', 'version', 'author', 'author_email', 'maintainer', 'maintainer_email', 'url', 'license', 'description', 'long_description', 'keywords', 'platforms', 'fullname', 'contact', 'contact_email', 'classifiers', 'download_url', 'provides', 'requires', 'obsoletes')

    def __init__(self, path=None):
        if path is not None:
            self.read_pkg_file(open(path))
        else:
            self.name = None
            self.version = None
            self.author = None
            self.author_email = None
            self.maintainer = None
            self.maintainer_email = None
            self.url = None
            self.license = None
            self.description = None
            self.long_description = None
            self.keywords = None
            self.platforms = None
            self.classifiers = None
            self.download_url = None
            self.provides = None
            self.requires = None
            self.obsoletes = None

    def read_pkg_file(self, file):
        """Reads the metadata values from a file object."""
        msg = message_from_file(file)

        def _read_field(name):
            value = msg[name]
            if value == 'UNKNOWN':
                return None
            return value

        def _read_list(name):
            values = msg.get_all(name, None)
            if values == []:
                return None
            return values
        metadata_version = msg['metadata-version']
        self.name = _read_field('name')
        self.version = _read_field('version')
        self.description = _read_field('summary')
        self.author = _read_field('author')
        self.maintainer = None
        self.author_email = _read_field('author-email')
        self.maintainer_email = None
        self.url = _read_field('home-page')
        self.license = _read_field('license')
        if 'download-url' in msg:
            self.download_url = _read_field('download-url')
        else:
            self.download_url = None
        self.long_description = _read_field('description')
        self.description = _read_field('summary')
        if 'keywords' in msg:
            self.keywords = _read_field('keywords').split(',')
        self.platforms = _read_list('platform')
        self.classifiers = _read_list('classifier')
        if metadata_version == '1.1':
            self.requires = _read_list('requires')
            self.provides = _read_list('provides')
            self.obsoletes = _read_list('obsoletes')
        else:
            self.requires = None
            self.provides = None
            self.obsoletes = None

    def write_pkg_info(self, base_dir):
        """Write the PKG-INFO file into the release tree.
        """
        with open(os.path.join(base_dir, 'PKG-INFO'), 'w', encoding='UTF-8') as pkg_info:
            self.write_pkg_file(pkg_info)

    def write_pkg_file(self, file):
        """Write the PKG-INFO format data to a file object.
        """
        version = '1.0'
        if self.provides or self.requires or self.obsoletes or self.classifiers or self.download_url:
            version = '1.1'
        file.write('Metadata-Version: %s\n' % version)
        file.write('Name: %s\n' % self.get_name())
        file.write('Version: %s\n' % self.get_version())
        file.write('Summary: %s\n' % self.get_description())
        file.write('Home-page: %s\n' % self.get_url())
        file.write('Author: %s\n' % self.get_contact())
        file.write('Author-email: %s\n' % self.get_contact_email())
        file.write('License: %s\n' % self.get_license())
        if self.download_url:
            file.write('Download-URL: %s\n' % self.download_url)
        long_desc = rfc822_escape(self.get_long_description())
        file.write('Description: %s\n' % long_desc)
        keywords = ','.join(self.get_keywords())
        if keywords:
            file.write('Keywords: %s\n' % keywords)
        self._write_list(file, 'Platform', self.get_platforms())
        self._write_list(file, 'Classifier', self.get_classifiers())
        self._write_list(file, 'Requires', self.get_requires())
        self._write_list(file, 'Provides', self.get_provides())
        self._write_list(file, 'Obsoletes', self.get_obsoletes())

    def _write_list(self, file, name, values):
        for value in values:
            file.write('%s: %s\n' % (name, value))

    def get_name(self):
        return self.name or 'UNKNOWN'

    def get_version(self):
        return self.version or '0.0.0'

    def get_fullname(self):
        return '%s-%s' % (self.get_name(), self.get_version())

    def get_author(self):
        return self.author or 'UNKNOWN'

    def get_author_email(self):
        return self.author_email or 'UNKNOWN'

    def get_maintainer(self):
        return self.maintainer or 'UNKNOWN'

    def get_maintainer_email(self):
        return self.maintainer_email or 'UNKNOWN'

    def get_contact(self):
        return self.maintainer or self.author or 'UNKNOWN'

    def get_contact_email(self):
        return self.maintainer_email or self.author_email or 'UNKNOWN'

    def get_url(self):
        return self.url or 'UNKNOWN'

    def get_license(self):
        return self.license or 'UNKNOWN'
    get_licence = get_license

    def get_description(self):
        return self.description or 'UNKNOWN'

    def get_long_description(self):
        return self.long_description or 'UNKNOWN'

    def get_keywords(self):
        return self.keywords or []

    def set_keywords(self, value):
        self.keywords = _ensure_list(value, 'keywords')

    def get_platforms(self):
        return self.platforms or ['UNKNOWN']

    def set_platforms(self, value):
        self.platforms = _ensure_list(value, 'platforms')

    def get_classifiers(self):
        return self.classifiers or []

    def set_classifiers(self, value):
        self.classifiers = _ensure_list(value, 'classifiers')

    def get_download_url(self):
        return self.download_url or 'UNKNOWN'

    def get_requires(self):
        return self.requires or []

    def set_requires(self, value):
        import distutils.versionpredicate
        for v in value:
            distutils.versionpredicate.VersionPredicate(v)
        self.requires = list(value)

    def get_provides(self):
        return self.provides or []

    def set_provides(self, value):
        value = [v.strip() for v in value]
        for v in value:
            import distutils.versionpredicate
            distutils.versionpredicate.split_provision(v)
        self.provides = value

    def get_obsoletes(self):
        return self.obsoletes or []

    def set_obsoletes(self, value):
        import distutils.versionpredicate
        for v in value:
            distutils.versionpredicate.VersionPredicate(v)
        self.obsoletes = list(value)