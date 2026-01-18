import time
import codecs
import sys
def _format_version_tuple(version_info):
    """Turn a version number 2, 3 or 5-tuple into a short string.

    This format matches <http://docs.python.org/dist/meta-data.html>
    and the typical presentation used in Python output.

    This also checks that the version is reasonable: the sub-release must be
    zero for final releases.

    >>> print(_format_version_tuple((1, 0, 0, 'final', 0)))
    1.0.0
    >>> print(_format_version_tuple((1, 2, 0, 'dev', 0)))
    1.2.0.dev
    >>> print(_format_version_tuple((1, 2, 0, 'dev', 1)))
    1.2.0.dev1
    >>> print(_format_version_tuple((1, 1, 1, 'candidate', 2)))
    1.1.1.rc2
    >>> print(_format_version_tuple((2, 1, 0, 'beta', 1)))
    2.1.b1
    >>> print(_format_version_tuple((1, 4, 0)))
    1.4.0
    >>> print(_format_version_tuple((1, 4)))
    1.4
    >>> print(_format_version_tuple((2, 1, 0, 'final', 42)))
    2.1.0.42
    >>> print(_format_version_tuple((1, 4, 0, 'wibble', 0)))
    1.4.0.wibble.0
    """
    if len(version_info) == 2:
        main_version = '%d.%d' % version_info[:2]
    else:
        main_version = '%d.%d.%d' % version_info[:3]
    if len(version_info) <= 3:
        return main_version
    release_type = version_info[3]
    sub = version_info[4]
    if release_type == 'final' and sub == 0:
        sub_string = ''
    elif release_type == 'final':
        sub_string = '.' + str(sub)
    elif release_type == 'dev' and sub == 0:
        sub_string = '.dev'
    elif release_type == 'dev':
        sub_string = '.dev' + str(sub)
    elif release_type in ('alpha', 'beta'):
        if version_info[2] == 0:
            main_version = '%d.%d' % version_info[:2]
        sub_string = '.' + release_type[0] + str(sub)
    elif release_type == 'candidate':
        sub_string = '.rc' + str(sub)
    else:
        return '.'.join(map(str, version_info))
    return main_version + sub_string