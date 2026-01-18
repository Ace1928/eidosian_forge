import gzip
import io
import tarfile
import sys
import os.path
from pathlib import Path
from debian.arfile import ArFile, ArError, ArMember     # pylint: disable=unused-import
from debian.changelog import Changelog
from debian.deb822 import Deb822
class DebPart(object):
    """'Part' of a .deb binary package.

    A .deb package is considered as made of 2 parts: a 'data' part
    (corresponding to the possibly compressed 'data.tar' archive embedded
    in a .deb) and a 'control' part (the 'control.tar.gz' archive). Each of
    them is represented by an instance of this class. Each archive should
    be a compressed tar archive although an uncompressed data.tar is permitted;
    supported compression formats are: .tar.gz, .tar.bz2, .tar.xz .

    When referring to file members of the underlying .tar.gz archive, file
    names can be specified in one of 3 formats "file", "./file", "/file". In
    all cases the file is considered relative to the root of the archive. For
    the control part the preferred mechanism is the first one (as in
    deb.control.get_content('control') ); for the data part the preferred
    mechanism is the third one (as in deb.data.get_file('/etc/vim/vimrc') ).
    """

    def __init__(self, member):
        self.__member = member
        self.__tgz = None

    def tgz(self):
        """Return a TarFile object corresponding to this part of a .deb
        package.

        Despite the name, this method gives access to various kind of
        compressed tar archives, not only gzipped ones.
        """

        def _custom_decompress(command_list):
            try:
                import subprocess
                import signal
                proc = subprocess.Popen(command_list, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=False, preexec_fn=lambda: signal.signal(signal.SIGPIPE, signal.SIG_DFL))
            except (OSError, ValueError) as e:
                raise DebError("error while running command '%s' as subprocess: '%s'" % (' '.join(command_list), e))
            data = proc.communicate(self.__member.read())[0]
            if proc.returncode != 0:
                raise DebError("command '%s' has failed with code '%s'" % (' '.join(command_list), proc.returncode))
            return io.BytesIO(data)
        if self.__tgz is None:
            name = self.__member.name
            extension = os.path.splitext(name)[1][1:]
            if extension in PART_EXTS or name == DATA_PART or name == CTRL_PART:
                if extension == 'zst':
                    buffer = _custom_decompress(['unzstd', '--stdout'])
                else:
                    buffer = self.__member
                try:
                    self.__tgz = tarfile.open(fileobj=buffer, mode='r:*')
                except (tarfile.ReadError, tarfile.CompressionError) as e:
                    raise DebError("tarfile has returned an error: '%s'" % e)
            else:
                raise DebError("part '%s' has unexpected extension" % name)
        return self.__tgz

    @staticmethod
    def __normalize_member(fname):
        """ try (not so hard) to obtain a member file name in a form that is
        stored in the .tar.gz, i.e. starting with ./ """
        fname = str(fname).replace('\\', '/')
        if fname.startswith('./'):
            return fname
        if fname.startswith('/'):
            return '.' + fname
        return './' + fname

    def __resolve_symlinks(self, path):
        """ walk the path following symlinks

        returns:
            resolved_path, info

        if the path is not found even after following symlinks within the
        archive, then None is returned.
        """
        try:
            resolved_path_parts = []
            for pathpart in path.split('/')[1:]:
                resolved_path_parts.append(pathpart)
                currpath = os.path.normpath('/'.join(resolved_path_parts))
                currpath = DebPart.__normalize_member(currpath)
                tinfo = self.tgz().getmember(currpath)
                if tinfo.issym():
                    if tinfo.linkname.startswith('/'):
                        resolved_path_parts = tinfo.linkname.split('/')
                        currpath = tinfo.linkname
                    else:
                        resolved_path_parts[-1] = tinfo.linkname
        except KeyError:
            return None
        return DebPart.__normalize_member(os.path.normpath(currpath))

    def has_file(self, fname, follow_symlinks=False):
        """Check if this part contains a given file name.

        Symlinks within the archive can be followed.
        """
        fname = DebPart.__normalize_member(fname)
        names = self.tgz().getnames()
        if fname in names:
            return True
        if follow_symlinks:
            fname_real = self.__resolve_symlinks(fname)
            return fname_real is not None
        return fname in names

    @overload
    def get_file(self, fname, encoding=None, errors=None, follow_symlinks=False):
        pass

    @overload
    def get_file(self, fname, encoding, errors=None, follow_symlinks=False):
        pass

    def get_file(self, fname, encoding=None, errors=None, follow_symlinks=False):
        """Return a file object corresponding to a given file name.

        If encoding is given, then the file object will return Unicode data;
        otherwise, it will return binary data.

        If follow_symlinks is True, then symlinks within the archive will be
        followed.
        """
        fname = DebPart.__normalize_member(fname)
        if follow_symlinks:
            fname_real = self.__resolve_symlinks(fname)
            if fname_real is None:
                raise DebError('File not found inside package')
            fname = fname_real
        try:
            fobj = self.tgz().extractfile(fname)
        except KeyError:
            raise DebError('File not found inside package')
        if fobj is None:
            raise DebError('File not found inside package')
        if encoding is not None:
            return io.TextIOWrapper(fobj, encoding=encoding, errors=errors)
        return fobj

    @overload
    def get_content(self, fname, encoding=None, errors=None, follow_symlinks=False):
        pass

    @overload
    def get_content(self, fname, encoding, errors=None, follow_symlinks=False):
        pass

    def get_content(self, fname, encoding=None, errors=None, follow_symlinks=False):
        """Return the string content of a given file, or None (e.g. for
        directories).

        If encoding is given, then the content will be a Unicode object;
        otherwise, it will contain binary data.

        If follow_symlinks is True, then symlinks within the archive will be
        followed.
        """
        f = self.get_file(str(fname), encoding=encoding, errors=errors, follow_symlinks=follow_symlinks)
        content = None
        if f:
            content = f.read()
            f.close()
        return content

    def __iter__(self):
        return iter(self.tgz().getnames())

    def __contains__(self, fname):
        return self.has_file(fname)

    def __getitem__(self, fname):
        return self.get_content(fname)

    def close(self):
        self.__member.close()