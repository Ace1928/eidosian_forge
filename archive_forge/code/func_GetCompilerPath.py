from xml.sax.saxutils import escape
import os.path
import subprocess
import gyp
import gyp.common
import gyp.msvs_emulation
import shlex
import xml.etree.cElementTree as ET
def GetCompilerPath(target_list, data, options):
    """Determine a command that can be used to invoke the compiler.

  Returns:
    If this is a gyp project that has explicit make settings, try to determine
    the compiler from that.  Otherwise, see if a compiler was specified via the
    CC_target environment variable.
  """
    build_file, _, _ = gyp.common.ParseQualifiedTarget(target_list[0])
    make_global_settings_dict = data[build_file].get('make_global_settings', {})
    for key, value in make_global_settings_dict:
        if key in ['CC', 'CXX']:
            return os.path.join(options.toplevel_dir, value)
    for key in ['CC_target', 'CC', 'CXX']:
        compiler = os.environ.get(key)
        if compiler:
            return compiler
    return 'gcc'