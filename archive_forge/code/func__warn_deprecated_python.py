import sys
import os
import errno
import socket
import warnings
from boto3.exceptions import PythonDeprecationWarning
import collections.abc as collections_abc
def _warn_deprecated_python():
    """Use this template for future deprecation campaigns as needed."""
    py_37_params = {'date': 'December 13, 2023', 'blog_link': 'https://aws.amazon.com/blogs/developer/python-support-policy-updates-for-aws-sdks-and-tools/'}
    deprecated_versions = {(3, 7): py_37_params}
    py_version = sys.version_info[:2]
    if py_version in deprecated_versions:
        params = deprecated_versions[py_version]
        warning = 'Boto3 will no longer support Python {}.{} starting {}. To continue receiving service updates, bug fixes, and security updates please upgrade to Python 3.8 or later. More information can be found here: {}'.format(py_version[0], py_version[1], params['date'], params['blog_link'])
        warnings.warn(warning, PythonDeprecationWarning)