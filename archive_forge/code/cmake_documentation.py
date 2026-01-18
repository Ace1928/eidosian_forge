import multiprocessing
import os
import signal
import subprocess
import gyp.common
import gyp.xcode_emulation
Converts Gyp target names into CMake target names.

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
  