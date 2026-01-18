from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import subprocess
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
def RequireJavaInstalled(for_text, min_version=7):
    """Require that a certain version of Java is installed.

  Args:
    for_text: str, the text explaining what Java is necessary for.
    min_version: int, the minimum major version to check for.

  Raises:
    JavaError: if a Java executable is not found or has the wrong version.

  Returns:
    str, Path to the Java executable.
  """
    java_path = files.FindExecutableOnPath('java')
    if not java_path:
        raise JavaError('To use the {for_text}, a Java {v}+ JRE must be installed and on your system PATH'.format(for_text=for_text, v=min_version))
    try:
        output = encoding.Decode(subprocess.check_output([java_path, '-Dfile.encoding=UTF-8', '-version'], stderr=subprocess.STDOUT), encoding='utf-8')
    except subprocess.CalledProcessError:
        raise JavaError('Unable to execute the java that was found on your PATH. The {for_text} requires a Java {v}+ JRE installed and on your system PATH'.format(for_text=for_text, v=min_version))
    java_exec_version_error = JavaVersionError('The java executable on your PATH is not a Java {v}+ JRE. The {for_text} requires a Java {v}+ JRE installed and on your system PATH'.format(v=min_version, for_text=for_text))
    match = re.search('version "1\\.', output)
    if match:
        match = re.search('version "(\\d+)\\.(\\d+)\\.', output)
        if not match:
            raise java_exec_version_error
        major_version = int(match.group(2))
    else:
        match = re.search('version "([1-9][0-9]*)', output)
        if not match:
            raise java_exec_version_error
        major_version = int(match.group(1))
    if major_version < min_version:
        raise java_exec_version_error
    return java_path