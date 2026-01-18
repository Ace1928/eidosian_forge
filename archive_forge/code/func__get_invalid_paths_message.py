import os
import sysconfig
def _get_invalid_paths_message(tzpaths):
    invalid_paths = (path for path in tzpaths if not os.path.isabs(path))
    prefix = '\n    '
    indented_str = prefix + prefix.join(invalid_paths)
    return 'Paths should be absolute but found the following relative paths:' + indented_str