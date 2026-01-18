import gzip
import io
import tarfile
import sys
import os.path
from pathlib import Path
from debian.arfile import ArFile, ArError, ArMember     # pylint: disable=unused-import
from debian.changelog import Changelog
from debian.deb822 import Deb822
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