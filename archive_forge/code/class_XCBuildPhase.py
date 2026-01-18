import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
class XCBuildPhase(XCObject):
    """Abstract base for build phase classes.  Not represented in a project
  file.

  Attributes:
    _files_by_path: A dict mapping each path of a child in the files list by
      path (keys) to the corresponding PBXBuildFile children (values).
    _files_by_xcfilelikeelement: A dict mapping each XCFileLikeElement (keys)
      to the corresponding PBXBuildFile children (values).
  """
    _schema = XCObject._schema.copy()
    _schema.update({'buildActionMask': [0, int, 0, 1, 2147483647], 'files': [1, PBXBuildFile, 1, 1, []], 'runOnlyForDeploymentPostprocessing': [0, int, 0, 1, 0]})

    def __init__(self, properties=None, id=None, parent=None):
        XCObject.__init__(self, properties, id, parent)
        self._files_by_path = {}
        self._files_by_xcfilelikeelement = {}
        for pbxbuildfile in self._properties.get('files', []):
            self._AddBuildFileToDicts(pbxbuildfile)

    def FileGroup(self, path):
        raise NotImplementedError(self.__class__.__name__ + ' must implement FileGroup')

    def _AddPathToDict(self, pbxbuildfile, path):
        """Adds path to the dict tracking paths belonging to this build phase.

    If the path is already a member of this build phase, raises an exception.
    """
        if path in self._files_by_path:
            raise ValueError('Found multiple build files with path ' + path)
        self._files_by_path[path] = pbxbuildfile

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

    def AppendBuildFile(self, pbxbuildfile, path=None):
        self.AppendProperty('files', pbxbuildfile)
        self._AddBuildFileToDicts(pbxbuildfile, path)

    def AddFile(self, path, settings=None):
        file_group, hierarchical = self.FileGroup(path)
        file_ref = file_group.AddOrGetFileByPath(path, hierarchical)
        if file_ref in self._files_by_xcfilelikeelement and isinstance(file_ref, PBXVariantGroup):
            pbxbuildfile = self._files_by_xcfilelikeelement[file_ref]
            self._AddBuildFileToDicts(pbxbuildfile, path)
        else:
            if settings is None:
                pbxbuildfile = PBXBuildFile({'fileRef': file_ref})
            else:
                pbxbuildfile = PBXBuildFile({'fileRef': file_ref, 'settings': settings})
            self.AppendBuildFile(pbxbuildfile, path)