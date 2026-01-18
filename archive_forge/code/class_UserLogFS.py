import typing
import abc
import six
from appdirs import AppDirs
from ._repr import make_repr
from .osfs import OSFS
class UserLogFS(_AppFS):
    """A filesystem for per-user application log data.

    May also be opened with
    ``open_fs('userlog://appname:author:version')``.

    """
    app_dir = 'user_log_dir'