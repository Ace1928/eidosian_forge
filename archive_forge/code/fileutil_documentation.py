import os
import posixpath
from typing import TYPE_CHECKING, Callable, Dict, Optional
from docutils.utils import relative_path
from sphinx.util.osutil import copyfile, ensuredir
from sphinx.util.typing import PathMatcher
Copy asset files to destination recursively.

    On copying, it expands the template variables if context argument is given and
    the asset is a template file.

    :param source: The path to source file or directory
    :param destination: The path to destination directory
    :param excluded: The matcher to determine the given path should be copied or not
    :param context: The template variables.  If not given, template files are simply copied
    :param renderer: The template engine.  If not given, SphinxRenderer is used by default
    :param onerror: The error handler.
    