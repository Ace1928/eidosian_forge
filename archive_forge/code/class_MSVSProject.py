import hashlib
import os
import random
from operator import attrgetter
import gyp.common
class MSVSProject(MSVSSolutionEntry):
    """Visual Studio project."""

    def __init__(self, path, name=None, dependencies=None, guid=None, spec=None, build_file=None, config_platform_overrides=None, fixpath_prefix=None):
        """Initializes the project.

    Args:
      path: Absolute path to the project file.
      name: Name of project.  If None, the name will be the same as the base
          name of the project file.
      dependencies: List of other Project objects this project is dependent
          upon, if not None.
      guid: GUID to use for project, if not None.
      spec: Dictionary specifying how to build this project.
      build_file: Filename of the .gyp file that the vcproj file comes from.
      config_platform_overrides: optional dict of configuration platforms to
          used in place of the default for this target.
      fixpath_prefix: the path used to adjust the behavior of _fixpath
    """
        self.path = path
        self.guid = guid
        self.spec = spec
        self.build_file = build_file
        self.name = name or os.path.splitext(os.path.basename(path))[0]
        self.dependencies = list(dependencies or [])
        self.entry_type_guid = ENTRY_TYPE_GUIDS['project']
        if config_platform_overrides:
            self.config_platform_overrides = config_platform_overrides
        else:
            self.config_platform_overrides = {}
        self.fixpath_prefix = fixpath_prefix
        self.msbuild_toolset = None

    def set_dependencies(self, dependencies):
        self.dependencies = list(dependencies or [])

    def get_guid(self):
        if self.guid is None:
            self.guid = MakeGuid(self.name)
        return self.guid

    def set_msbuild_toolset(self, msbuild_toolset):
        self.msbuild_toolset = msbuild_toolset