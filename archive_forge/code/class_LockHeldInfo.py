import os
import time
import yaml
from . import config, debug, errors, lock, osutils, ui, urlutils
from .decorators import only_raises
from .errors import (DirectoryNotEmpty, LockBreakMismatch, LockBroken,
from .i18n import gettext
from .osutils import format_delta, get_host_name, rand_chars
from .trace import mutter, note
from .transport import FileExists, NoSuchFile
class LockHeldInfo:
    """The information recorded about a held lock.

    This information is recorded into the lock when it's taken, and it can be
    read back by any process with access to the lockdir.  It can be used, for
    example, to tell the user who holds the lock, or to try to detect whether
    the lock holder is still alive.
    """

    def __init__(self, info_dict):
        self.info_dict = info_dict

    def __repr__(self):
        """Return a debugging representation of this object."""
        return '{}({!r})'.format(self.__class__.__name__, self.info_dict)

    def __str__(self):
        """Return a user-oriented description of this object."""
        d = self.to_readable_dict()
        return gettext('held by %(user)s on %(hostname)s (process #%(pid)s), acquired %(time_ago)s') % d

    def to_readable_dict(self):
        """Turn the holder info into a dict of human-readable attributes.

        For example, the start time is presented relative to the current time,
        rather than as seconds since the epoch.

        Returns a list of [user, hostname, pid, time_ago] all as readable
        strings.
        """
        start_time = self.info_dict.get('start_time')
        if start_time is None:
            time_ago = '(unknown)'
        else:
            time_ago = format_delta(time.time() - self.info_dict['start_time'])
        user = self.info_dict.get('user', '<unknown>')
        hostname = self.info_dict.get('hostname', '<unknown>')
        pid = self.info_dict.get('pid', '<unknown>')
        return dict(user=user, hostname=hostname, pid=pid, time_ago=time_ago)

    @property
    def nonce(self):
        nonce = self.get('nonce')
        return nonce.encode('ascii') if nonce else None

    def get(self, field_name):
        """Return the contents of a field from the lock info, or None."""
        return self.info_dict.get(field_name)

    @classmethod
    def for_this_process(cls, extra_holder_info):
        """Return a new LockHeldInfo for a lock taken by this process.
        """
        info = dict(hostname=get_host_name(), pid=os.getpid(), nonce=rand_chars(20), start_time=int(time.time()), user=get_username_for_lock_info())
        if extra_holder_info is not None:
            info.update(extra_holder_info)
        return cls(info)

    def to_bytes(self):
        return yaml.dump(self.info_dict).encode('utf-8')

    @classmethod
    def from_info_file_bytes(cls, info_file_bytes):
        """Construct from the contents of the held file."""
        try:
            ret = yaml.safe_load(info_file_bytes)
        except yaml.reader.ReaderError as e:
            lines = osutils.split_lines(info_file_bytes)
            mutter('Corrupt lock info file: %r', lines)
            raise LockCorrupt('could not parse lock info file: ' + str(e), lines)
        if ret is None:
            return cls({})
        else:
            return cls(ret)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        """Equality check for lock holders."""
        if type(self) != type(other):
            return False
        return self.info_dict == other.info_dict

    def __ne__(self, other):
        return not self == other

    def is_locked_by_this_process(self):
        """True if this process seems to be the current lock holder."""
        return self.get('hostname') == get_host_name() and self.get('pid') == os.getpid() and (self.get('user') == get_username_for_lock_info())

    def is_lock_holder_known_dead(self):
        """True if the lock holder process is known to be dead.

        False if it's either known to be still alive, or if we just can't tell.

        We can be fairly sure the lock holder is dead if it declared the same
        hostname and there is no process with the given pid alive.  If people
        have multiple machines with the same hostname this may cause trouble.

        This doesn't check whether the lock holder is in fact the same process
        calling this method.  (In that case it will return true.)
        """
        if self.get('hostname') != get_host_name():
            return False
        if self.get('hostname') == 'localhost':
            return False
        if self.get('user') != get_username_for_lock_info():
            return False
        pid_str = self.info_dict.get('pid', None)
        if not pid_str:
            mutter('no pid recorded in {!r}'.format(self))
            return False
        try:
            pid = int(pid_str)
        except ValueError:
            mutter("can't parse pid %r from %r" % (pid_str, self))
            return False
        return osutils.is_local_pid_dead(pid)