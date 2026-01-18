from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core.util import encoding
def RemovePrivateInformationFromTraceback(traceback):
    """Given a stacktrace, only include Cloud SDK files in path.

  Args:
    traceback: str, the original unformatted traceback

  Returns:
    str, A new stacktrace with the private paths removed
    None, If traceback does not match traceback pattern
  """
    match = re.search(PARTITION_TRACEBACK_PATTERN, traceback)
    if not match:
        return None
    stacktrace_list = traceback.splitlines()
    remove_path_stacktrace_list = []
    pattern_file_path = re.compile(FILE_PATH_PATTERN)
    lines_since_match = 0
    for line in stacktrace_list:
        match = pattern_file_path.match(line)
        if match:
            remove_path_stacktrace_list.append(line.replace(match.group('file'), ''))
            lines_since_match = 0
        else:
            lines_since_match += 1
            if lines_since_match > 2:
                remove_path_stacktrace_list[-1] += line
            else:
                remove_path_stacktrace_list.append(line)
    exception_line = remove_path_stacktrace_list[-1]
    exception_line = encoding.Decode(exception_line).split(':', 1)[0]
    remove_path_stacktrace_list[-1] = exception_line
    formatted_stacktrace = '\n'.join((line for line in remove_path_stacktrace_list)) + '\n'
    return formatted_stacktrace