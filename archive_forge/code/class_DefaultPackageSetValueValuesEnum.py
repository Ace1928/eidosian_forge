from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DefaultPackageSetValueValuesEnum(_messages.Enum):
    """The default package set to install. This allows the service to select
    a default set of packages which are useful to worker harnesses written in
    a particular language.

    Values:
      DEFAULT_PACKAGE_SET_UNKNOWN: The default set of packages to stage is
        unknown, or unspecified.
      DEFAULT_PACKAGE_SET_NONE: Indicates that no packages should be staged at
        the worker unless explicitly specified by the job.
      DEFAULT_PACKAGE_SET_JAVA: Stage packages typically useful to workers
        written in Java.
      DEFAULT_PACKAGE_SET_PYTHON: Stage packages typically useful to workers
        written in Python.
    """
    DEFAULT_PACKAGE_SET_UNKNOWN = 0
    DEFAULT_PACKAGE_SET_NONE = 1
    DEFAULT_PACKAGE_SET_JAVA = 2
    DEFAULT_PACKAGE_SET_PYTHON = 3