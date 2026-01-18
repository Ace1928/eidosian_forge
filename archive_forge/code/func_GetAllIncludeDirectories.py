from xml.sax.saxutils import escape
import os.path
import subprocess
import gyp
import gyp.common
import gyp.msvs_emulation
import shlex
import xml.etree.cElementTree as ET
def GetAllIncludeDirectories(target_list, target_dicts, shared_intermediate_dirs, config_name, params, compiler_path):
    """Calculate the set of include directories to be used.

  Returns:
    A list including all the include_dir's specified for every target followed
    by any include directories that were added as cflag compiler options.
  """
    gyp_includes_set = set()
    compiler_includes_list = []
    if compiler_path:
        command = shlex.split(compiler_path)
        command.extend(['-E', '-xc++', '-v', '-'])
        proc = subprocess.Popen(args=command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = proc.communicate()[1].decode('utf-8')
        in_include_list = False
        for line in output.splitlines():
            if line.startswith('#include'):
                in_include_list = True
                continue
            if line.startswith('End of search list.'):
                break
            if in_include_list:
                include_dir = line.strip()
                if include_dir not in compiler_includes_list:
                    compiler_includes_list.append(include_dir)
    flavor = gyp.common.GetFlavor(params)
    if flavor == 'win':
        generator_flags = params.get('generator_flags', {})
    for target_name in target_list:
        target = target_dicts[target_name]
        if config_name in target['configurations']:
            config = target['configurations'][config_name]
            if flavor == 'win':
                msvs_settings = gyp.msvs_emulation.MsvsSettings(target, generator_flags)
                cflags = msvs_settings.GetCflags(config_name)
            else:
                cflags = config['cflags']
            for cflag in cflags:
                if cflag.startswith('-I'):
                    include_dir = cflag[2:]
                    if include_dir not in compiler_includes_list:
                        compiler_includes_list.append(include_dir)
            if 'include_dirs' in config:
                include_dirs = config['include_dirs']
                for shared_intermediate_dir in shared_intermediate_dirs:
                    for include_dir in include_dirs:
                        include_dir = include_dir.replace('$SHARED_INTERMEDIATE_DIR', shared_intermediate_dir)
                        if not os.path.isabs(include_dir):
                            base_dir = os.path.dirname(target_name)
                            include_dir = base_dir + '/' + include_dir
                            include_dir = os.path.abspath(include_dir)
                        gyp_includes_set.add(include_dir)
    all_includes_list = list(gyp_includes_set)
    all_includes_list.sort()
    for compiler_include in compiler_includes_list:
        if compiler_include not in gyp_includes_set:
            all_includes_list.append(compiler_include)
    return all_includes_list