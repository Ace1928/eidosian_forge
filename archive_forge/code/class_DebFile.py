import gzip
import io
import tarfile
import sys
import os.path
from pathlib import Path
from debian.arfile import ArFile, ArError, ArMember     # pylint: disable=unused-import
from debian.changelog import Changelog
from debian.deb822 import Deb822
class DebFile(ArFile):
    """Representation of a .deb file (a Debian binary package)

    DebFile objects have the following (read-only) properties:
        - version       debian .deb file format version (not related with the
                        contained package version), 2.0 at the time of writing
                        for all .deb packages in the Debian archive
        - data          DebPart object corresponding to the data.tar.gz (or
                        other compressed or uncompressed tar) archive contained
                        in the .deb file
        - control       DebPart object corresponding to the control.tar.gz (or
                        other compressed tar) archive contained in the .deb
                        file
    """

    def __init__(self, filename=None, mode='r', fileobj=None):
        ArFile.__init__(self, filename, mode, fileobj)
        actual_names = set(self.getnames())

        def compressed_part_name(basename):
            candidates = ['%s.%s' % (basename, ext) for ext in PART_EXTS]
            if basename in (DATA_PART, CTRL_PART):
                candidates.append(basename)
            parts = actual_names.intersection(set(candidates))
            if not parts:
                raise DebError('missing required part in given .deb (expected one of: %s)' % candidates)
            if len(parts) > 1:
                raise DebError('too many parts in given .deb (was looking for only one of: %s)' % candidates)
            return list(parts)[0]
        if INFO_PART not in actual_names:
            raise DebError("missing required part in given .deb (expected: '%s')" % INFO_PART)
        self.__parts = {}
        self.__parts[CTRL_PART] = DebControl(self.getmember(compressed_part_name(CTRL_PART)))
        self.__parts[DATA_PART] = DebData(self.getmember(compressed_part_name(DATA_PART)))
        self.__pkgname = None
        f = self.getmember(INFO_PART)
        self.__version = f.read().strip()
        f.close()

    def __updatePkgName(self):
        self.__pkgname = self.debcontrol()['package']

    @property
    def version(self):
        return self.__version

    @property
    def data(self):
        return self.__parts[DATA_PART]

    @property
    def control(self):
        return self.__parts[CTRL_PART]

    def debcontrol(self):
        """ See .control.debcontrol() """
        return self.control.debcontrol()

    def scripts(self):
        """ See .control.scripts() """
        return self.control.scripts()

    @overload
    def md5sums(self, encoding=None, errors=None):
        pass

    @overload
    def md5sums(self, encoding, errors=None):
        pass

    def md5sums(self, encoding=None, errors=None):
        """ See .control.md5sums() """
        return self.control.md5sums(encoding=encoding, errors=errors)

    def changelog(self):
        """ Return a Changelog object for the changelog.Debian.gz of the
        present .deb package. Return None if no changelog can be found. """
        if self.__pkgname is None:
            self.__updatePkgName()
        for fname in [CHANGELOG_DEBIAN % self.__pkgname, CHANGELOG_NATIVE % self.__pkgname]:
            try:
                fh = self.data.get_file(fname, follow_symlinks=True)
            except DebError:
                continue
            with gzip.GzipFile(fileobj=fh) as gz:
                raw_changelog = gz.read()
            return Changelog(raw_changelog)
        return None

    def close(self):
        self.control.close()
        self.data.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()