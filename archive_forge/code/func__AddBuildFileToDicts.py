import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def _AddBuildFileToDicts(self, pbxbuildfile, path=None):
    """Maintains the _files_by_path and _files_by_xcfilelikeelement dicts.

    If path is specified, then it is the path that is being added to the
    phase, and pbxbuildfile must contain either a PBXFileReference directly
    referencing that path, or it must contain a PBXVariantGroup that itself
    contains a PBXFileReference referencing the path.

    If path is not specified, either the PBXFileReference's path or the paths
    of all children of the PBXVariantGroup are taken as being added to the
    phase.

    If the path is already present in the phase, raises an exception.

    If the PBXFileReference or PBXVariantGroup referenced by pbxbuildfile
    are already present in the phase, referenced by a different PBXBuildFile
    object, raises an exception.  This does not raise an exception when
    a PBXFileReference or PBXVariantGroup reappear and are referenced by the
    same PBXBuildFile that has already introduced them, because in the case
    of PBXVariantGroup objects, they may correspond to multiple paths that are
    not all added simultaneously.  When this situation occurs, the path needs
    to be added to _files_by_path, but nothing needs to change in
    _files_by_xcfilelikeelement, and the caller should have avoided adding
    the PBXBuildFile if it is already present in the list of children.
    """
    xcfilelikeelement = pbxbuildfile._properties['fileRef']
    paths = []
    if path is not None:
        if isinstance(xcfilelikeelement, PBXVariantGroup):
            paths.append(path)
    elif isinstance(xcfilelikeelement, PBXVariantGroup):
        for variant in xcfilelikeelement._properties['children']:
            paths.append(variant.FullPath())
    else:
        paths.append(xcfilelikeelement.FullPath())
    for a_path in paths:
        self._AddPathToDict(pbxbuildfile, a_path)
    if xcfilelikeelement in self._files_by_xcfilelikeelement and self._files_by_xcfilelikeelement[xcfilelikeelement] != pbxbuildfile:
        raise ValueError('Found multiple build files for ' + xcfilelikeelement.Name())
    self._files_by_xcfilelikeelement[xcfilelikeelement] = pbxbuildfile