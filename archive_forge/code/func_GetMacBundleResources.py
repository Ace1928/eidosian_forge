import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetMacBundleResources(product_dir, xcode_settings, resources):
    """Yields (output, resource) pairs for every resource in |resources|.
  Only call this for mac bundle targets.

  Args:
      product_dir: Path to the directory containing the output bundle,
          relative to the build directory.
      xcode_settings: The XcodeSettings of the current target.
      resources: A list of bundle resources, relative to the build directory.
  """
    dest = os.path.join(product_dir, xcode_settings.GetBundleResourceFolder())
    for res in resources:
        output = dest
        assert ' ' not in res, 'Spaces in resource filenames not supported (%s)' % res
        res_parts = os.path.split(res)
        lproj_parts = os.path.split(res_parts[0])
        if lproj_parts[1].endswith('.lproj'):
            output = os.path.join(output, lproj_parts[1])
        output = os.path.join(output, res_parts[1])
        if output.endswith('.xib'):
            output = os.path.splitext(output)[0] + '.nib'
        if output.endswith('.storyboard'):
            output = os.path.splitext(output)[0] + '.storyboardc'
        yield (output, res)