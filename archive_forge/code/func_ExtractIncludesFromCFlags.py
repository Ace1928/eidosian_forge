import gyp
import gyp.common
import gyp.generator.make as make  # Reuse global functions from make backend.
import os
import re
import subprocess
def ExtractIncludesFromCFlags(self, cflags):
    """Extract includes "-I..." out from cflags

        Args:
          cflags: A list of compiler flags, which may be mixed with "-I.."
        Returns:
          A tuple of lists: (clean_clfags, include_paths). "-I.." is trimmed.
        """
    clean_cflags = []
    include_paths = []
    for flag in cflags:
        if flag.startswith('-I'):
            include_paths.append(flag[2:])
        else:
            clean_cflags.append(flag)
    return (clean_cflags, include_paths)