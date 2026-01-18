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
class LockDir(lock.Lock):
    """Write-lock guarding access to data.
    """
    __INFO_NAME = '/info'

    def __init__(self, transport, path, file_modebits=420, dir_modebits=493, extra_holder_info=None):
        """Create a new LockDir object.

        The LockDir is initially unlocked - this just creates the object.

        :param transport: Transport which will contain the lock

        :param path: Path to the lock within the base directory of the
            transport.

        :param extra_holder_info: If passed, {str:str} dict of extra or
            updated information to insert into the info file when the lock is
            taken.
        """
        self.transport = transport
        self.path = path
        self._lock_held = False
        self._locked_via_token = False
        self._fake_read_lock = False
        self._held_dir = path + '/held'
        self._held_info_path = self._held_dir + self.__INFO_NAME
        self._file_modebits = file_modebits
        self._dir_modebits = dir_modebits
        self._report_function = note
        self.extra_holder_info = extra_holder_info
        self._warned_about_lock_holder = None

    def __repr__(self):
        return '{}({}{})'.format(self.__class__.__name__, self.transport.base, self.path)
    is_held = property(lambda self: self._lock_held)

    def create(self, mode=None):
        """Create the on-disk lock.

        This is typically only called when the object/directory containing the
        directory is first created.  The lock is not held when it's created.
        """
        self._trace('create lock directory')
        try:
            self.transport.mkdir(self.path, mode=mode)
        except (TransportError, PathError) as e:
            raise LockFailed(self, e)

    def _attempt_lock(self):
        """Make the pending directory and attempt to rename into place.

        If the rename succeeds, we read back the info file to check that we
        really got the lock.

        If we fail to acquire the lock, this method is responsible for
        cleaning up the pending directory if possible.  (But it doesn't do
        that yet.)

        :returns: The nonce of the lock, if it was successfully acquired.

        :raises LockContention: If the lock is held by someone else.  The
            exception contains the info of the current holder of the lock.
        """
        self._trace('lock_write...')
        start_time = time.time()
        try:
            tmpname = self._create_pending_dir()
        except (errors.TransportError, PathError) as e:
            self._trace('... failed to create pending dir, %s', e)
            raise LockFailed(self, e)
        while True:
            try:
                self.transport.rename(tmpname, self._held_dir)
                break
            except (errors.TransportError, PathError, DirectoryNotEmpty, FileExists, ResourceBusy) as e:
                self._trace('... contention, %s', e)
                other_holder = self.peek()
                self._trace('other holder is %r' % other_holder)
                try:
                    self._handle_lock_contention(other_holder)
                except BaseException:
                    self._remove_pending_dir(tmpname)
                    raise
            except Exception as e:
                self._trace('... lock failed, %s', e)
                self._remove_pending_dir(tmpname)
                raise
        info = self.peek()
        self._trace('after locking, info=%r', info)
        if info is None:
            raise LockFailed(self, 'lock was renamed into place, but now is missing!')
        if info.nonce != self.nonce:
            self._trace('rename succeeded, but lock is still held by someone else')
            raise LockContention(self)
        self._lock_held = True
        self._trace('... lock succeeded after %dms', (time.time() - start_time) * 1000)
        return self.nonce

    def _handle_lock_contention(self, other_holder):
        """A lock we want to take is held by someone else.

        This function can: tell the user about it; possibly detect that it's
        safe or appropriate to steal the lock, or just raise an exception.

        If this function returns (without raising an exception) the lock will
        be attempted again.

        :param other_holder: A LockHeldInfo for the current holder; note that
            it might be None if the lock can be seen to be held but the info
            can't be read.
        """
        if other_holder is not None:
            if other_holder.is_lock_holder_known_dead():
                if self.get_config().get('locks.steal_dead'):
                    ui.ui_factory.show_user_warning('locks_steal_dead', lock_url=urlutils.join(self.transport.base, self.path), other_holder_info=str(other_holder))
                    self.force_break(other_holder)
                    self._trace('stole lock from dead holder')
                    return
        raise LockContention(self)

    def _remove_pending_dir(self, tmpname):
        """Remove the pending directory

        This is called if we failed to rename into place, so that the pending
        dirs don't clutter up the lockdir.
        """
        self._trace('remove %s', tmpname)
        try:
            self.transport.delete(tmpname + self.__INFO_NAME)
            self.transport.rmdir(tmpname)
        except PathError as e:
            note(gettext('error removing pending lock: %s'), e)

    def _create_pending_dir(self):
        tmpname = '{}/{}.tmp'.format(self.path, rand_chars(10))
        try:
            self.transport.mkdir(tmpname)
        except NoSuchFile:
            self._trace('lock directory does not exist, creating it')
            self.create(mode=self._dir_modebits)
            self.transport.mkdir(tmpname)
        info = LockHeldInfo.for_this_process(self.extra_holder_info)
        self.nonce = info.nonce
        self.transport.put_bytes_non_atomic(tmpname + self.__INFO_NAME, info.to_bytes())
        return tmpname

    @only_raises(LockNotHeld, LockBroken)
    def unlock(self):
        """Release a held lock
        """
        if self._fake_read_lock:
            self._fake_read_lock = False
            return
        if not self._lock_held:
            return lock.cant_unlock_not_held(self)
        if self._locked_via_token:
            self._locked_via_token = False
            self._lock_held = False
        else:
            old_nonce = self.nonce
            start_time = time.time()
            self._trace('unlocking')
            tmpname = '{}/releasing.{}.tmp'.format(self.path, rand_chars(20))
            self.confirm()
            self.transport.rename(self._held_dir, tmpname)
            self._lock_held = False
            self.transport.delete(tmpname + self.__INFO_NAME)
            try:
                self.transport.rmdir(tmpname)
            except DirectoryNotEmpty:
                self._trace('doing recursive deletion of non-empty directory %s', tmpname)
                self.transport.delete_tree(tmpname)
            self._trace('... unlock succeeded after %dms', (time.time() - start_time) * 1000)
            result = lock.LockResult(self.transport.abspath(self.path), old_nonce)
            for hook in self.hooks['lock_released']:
                hook(result)

    def break_lock(self):
        """Break a lock not held by this instance of LockDir.

        This is a UI centric function: it uses the ui.ui_factory to
        prompt for input if a lock is detected and there is any doubt about
        it possibly being still active.  force_break is the non-interactive
        version.

        :returns: LockResult for the broken lock.
        """
        self._check_not_locked()
        try:
            holder_info = self.peek()
        except LockCorrupt as e:
            if ui.ui_factory.get_boolean('Break (corrupt {!r})'.format(self)):
                self.force_break_corrupt(e.file_data)
            return
        if holder_info is not None:
            if ui.ui_factory.confirm_action('Break %(lock_info)s', 'breezy.lockdir.break', dict(lock_info=str(holder_info))):
                result = self.force_break(holder_info)
                ui.ui_factory.show_message('Broke lock %s' % result.lock_url)

    def force_break(self, dead_holder_info):
        """Release a lock held by another process.

        WARNING: This should only be used when the other process is dead; if
        it still thinks it has the lock there will be two concurrent writers.
        In general the user's approval should be sought for lock breaks.

        After the lock is broken it will not be held by any process.
        It is possible that another process may sneak in and take the
        lock before the breaking process acquires it.

        :param dead_holder_info:
            Must be the result of a previous LockDir.peek() call; this is used
            to check that it's still held by the same process that the user
            decided was dead.  If this is not the current holder,
            LockBreakMismatch is raised.

        :returns: LockResult for the broken lock.
        """
        if not isinstance(dead_holder_info, LockHeldInfo):
            raise ValueError('dead_holder_info: %r' % dead_holder_info)
        self._check_not_locked()
        current_info = self.peek()
        if current_info is None:
            return
        if current_info != dead_holder_info:
            raise LockBreakMismatch(self, current_info, dead_holder_info)
        tmpname = '{}/broken.{}.tmp'.format(self.path, rand_chars(20))
        self.transport.rename(self._held_dir, tmpname)
        broken_info_path = tmpname + self.__INFO_NAME
        broken_info = self._read_info_file(broken_info_path)
        if broken_info != dead_holder_info:
            raise LockBreakMismatch(self, broken_info, dead_holder_info)
        self.transport.delete(broken_info_path)
        self.transport.rmdir(tmpname)
        result = lock.LockResult(self.transport.abspath(self.path), current_info.nonce)
        for hook in self.hooks['lock_broken']:
            hook(result)
        return result

    def force_break_corrupt(self, corrupt_info_lines):
        """Release a lock that has been corrupted.

        This is very similar to force_break, it except it doesn't assume that
        self.peek() can work.

        :param corrupt_info_lines: the lines of the corrupted info file, used
            to check that the lock hasn't changed between reading the (corrupt)
            info file and calling force_break_corrupt.
        """
        self._check_not_locked()
        tmpname = '{}/broken.{}.tmp'.format(self.path, rand_chars(20))
        self.transport.rename(self._held_dir, tmpname)
        broken_info_path = tmpname + self.__INFO_NAME
        broken_content = self.transport.get_bytes(broken_info_path)
        broken_lines = osutils.split_lines(broken_content)
        if broken_lines != corrupt_info_lines:
            raise LockBreakMismatch(self, broken_lines, corrupt_info_lines)
        self.transport.delete(broken_info_path)
        self.transport.rmdir(tmpname)
        result = lock.LockResult(self.transport.abspath(self.path))
        for hook in self.hooks['lock_broken']:
            hook(result)

    def _check_not_locked(self):
        """If the lock is held by this instance, raise an error."""
        if self._lock_held:
            raise AssertionError("can't break own lock: %r" % self)

    def confirm(self):
        """Make sure that the lock is still held by this locker.

        This should only fail if the lock was broken by user intervention,
        or if the lock has been affected by a bug.

        If the lock is not thought to be held, raises LockNotHeld.  If
        the lock is thought to be held but has been broken, raises
        LockBroken.
        """
        if not self._lock_held:
            raise LockNotHeld(self)
        info = self.peek()
        if info is None:
            raise LockBroken(self)
        if info.nonce != self.nonce:
            raise LockBroken(self)

    def _read_info_file(self, path):
        """Read one given info file.

        peek() reads the info file of the lock holder, if any.
        """
        return LockHeldInfo.from_info_file_bytes(self.transport.get_bytes(path))

    def peek(self):
        """Check if the lock is held by anyone.

        If it is held, this returns the lock info structure as a dict
        which contains some information about the current lock holder.
        Otherwise returns None.
        """
        try:
            info = self._read_info_file(self._held_info_path)
            self._trace('peek -> held')
            return info
        except NoSuchFile:
            self._trace('peek -> not held')

    def _prepare_info(self):
        """Write information about a pending lock to a temporary file.
        """

    def attempt_lock(self):
        """Take the lock; fail if it's already held.

        If you wish to block until the lock can be obtained, call wait_lock()
        instead.

        :return: The lock token.
        :raises LockContention: if the lock is held by someone else.
        """
        if self._fake_read_lock:
            raise LockContention(self)
        result = self._attempt_lock()
        hook_result = lock.LockResult(self.transport.abspath(self.path), self.nonce)
        for hook in self.hooks['lock_acquired']:
            hook(hook_result)
        return result

    def lock_url_for_display(self):
        """Give a nicely-printable representation of the URL of this lock."""
        lock_url = self.transport.abspath(self.path)
        if lock_url.startswith('file://'):
            lock_url = lock_url.split('.bzr/')[0]
        else:
            lock_url = ''
        return lock_url

    def wait_lock(self, timeout=None, poll=None, max_attempts=None):
        """Wait a certain period for a lock.

        If the lock can be acquired within the bounded time, it
        is taken and this returns.  Otherwise, LockContention
        is raised.  Either way, this function should return within
        approximately `timeout` seconds.  (It may be a bit more if
        a transport operation takes a long time to complete.)

        :param timeout: Approximate maximum amount of time to wait for the
        lock, in seconds.

        :param poll: Delay in seconds between retrying the lock.

        :param max_attempts: Maximum number of times to try to lock.

        :return: The lock token.
        """
        if timeout is None:
            timeout = _DEFAULT_TIMEOUT_SECONDS
        if poll is None:
            poll = _DEFAULT_POLL_SECONDS
        deadline = time.time() + timeout
        deadline_str = None
        last_info = None
        attempt_count = 0
        lock_url = self.lock_url_for_display()
        while True:
            attempt_count += 1
            try:
                return self.attempt_lock()
            except LockContention:
                pass
            new_info = self.peek()
            if new_info is not None and new_info != last_info:
                if last_info is None:
                    start = gettext('Unable to obtain')
                else:
                    start = gettext('Lock owner changed for')
                last_info = new_info
                msg = gettext('{0} lock {1} {2}.').format(start, lock_url, new_info)
                if deadline_str is None:
                    deadline_str = time.strftime('%H:%M:%S', time.localtime(deadline))
                if timeout > 0:
                    msg += '\n' + gettext('Will continue to try until %s, unless you press Ctrl-C.') % deadline_str
                msg += '\n' + gettext('See "brz help break-lock" for more.')
                self._report_function(msg)
            if max_attempts is not None and attempt_count >= max_attempts:
                self._trace('exceeded %d attempts')
                raise LockContention(self)
            if time.time() + poll < deadline:
                self._trace('waiting %ss', poll)
                time.sleep(poll)
            else:
                self._trace('timeout after waiting %ss', timeout)
                raise LockContention('(local)', lock_url)

    def leave_in_place(self):
        self._locked_via_token = True

    def dont_leave_in_place(self):
        self._locked_via_token = False

    def lock_write(self, token=None):
        """Wait for and acquire the lock.

        :param token: if this is already locked, then lock_write will fail
            unless the token matches the existing lock.
        :returns: a token if this instance supports tokens, otherwise None.
        :raises TokenLockingNotSupported: when a token is given but this
            instance doesn't support using token locks.
        :raises MismatchedToken: if the specified token doesn't match the token
            of the existing lock.

        A token should be passed in if you know that you have locked the object
        some other way, and need to synchronise this object's state with that
        fact.

        XXX: docstring duplicated from LockableFiles.lock_write.
        """
        if token is not None:
            self.validate_token(token)
            self.nonce = token
            self._lock_held = True
            self._locked_via_token = True
            return token
        else:
            return self.wait_lock()

    def lock_read(self):
        """Compatibility-mode shared lock.

        LockDir doesn't support shared read-only locks, so this
        just pretends that the lock is taken but really does nothing.
        """
        if self._lock_held or self._fake_read_lock:
            raise LockContention(self)
        self._fake_read_lock = True

    def validate_token(self, token):
        if token is not None:
            info = self.peek()
            if info is None:
                lock_token = None
            else:
                lock_token = info.nonce
            if token != lock_token:
                raise errors.TokenMismatch(token, lock_token)
            else:
                self._trace('revalidated by token %r', token)

    def _trace(self, format, *args):
        if 'lock' not in debug.debug_flags:
            return
        mutter(str(self) + ': ' + format % args)

    def get_config(self):
        """Get the configuration that governs this lockdir."""
        return config.GlobalStack()