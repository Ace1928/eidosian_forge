import multiprocessing
import os
import signal
import subprocess
import gyp.common
import gyp.xcode_emulation
class CMakeNamer:
    """Converts Gyp target names into CMake target names.

  CMake requires that target names be globally unique. One way to ensure
  this is to fully qualify the names of the targets. Unfortunately, this
  ends up with all targets looking like "chrome_chrome_gyp_chrome" instead
  of just "chrome". If this generator were only interested in building, it
  would be possible to fully qualify all target names, then create
  unqualified target names which depend on all qualified targets which
  should have had that name. This is more or less what the 'make' generator
  does with aliases. However, one goal of this generator is to create CMake
  files for use with IDEs, and fully qualified names are not as user
  friendly.

  Since target name collision is rare, we do the above only when required.

  Toolset variants are always qualified from the base, as this is required for
  building. However, it also makes sense for an IDE, as it is possible for
  defines to be different.
  """

    def __init__(self, target_list):
        self.cmake_target_base_names_conficting = set()
        cmake_target_base_names_seen = set()
        for qualified_target in target_list:
            cmake_target_base_name = CreateCMakeTargetBaseName(qualified_target)
            if cmake_target_base_name not in cmake_target_base_names_seen:
                cmake_target_base_names_seen.add(cmake_target_base_name)
            else:
                self.cmake_target_base_names_conficting.add(cmake_target_base_name)

    def CreateCMakeTargetName(self, qualified_target):
        base_name = CreateCMakeTargetBaseName(qualified_target)
        if base_name in self.cmake_target_base_names_conficting:
            return CreateCMakeTargetFullName(qualified_target)
        return base_name