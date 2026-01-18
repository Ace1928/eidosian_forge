from .. import errors
from ..decorators import only_raises
class LockWrapper:
    """A wrapper which lets us set locking ability.

    This also lets us record what objects were locked in what order,
    to ensure that locking happens correctly.
    """

    def __init__(self, sequence, other, other_id):
        """Wrap a locking policy around a given object.

        :param sequence: A list object where we should record actions
        :param other: The object to control policy on
        :param other_id: Something to identify the object by
        """
        self.__dict__['_sequence'] = sequence
        self.__dict__['_other'] = other
        self.__dict__['_other_id'] = other_id
        self.__dict__['_allow_write'] = True
        self.__dict__['_allow_read'] = True
        self.__dict__['_allow_unlock'] = True

    def __eq__(self, other):
        if isinstance(other, LockWrapper):
            return self._other == other._other
        return False

    def __getattr__(self, attr):
        return getattr(self._other, attr)

    def __setattr__(self, attr, val):
        return setattr(self._other, attr, val)

    def lock_read(self):
        self._sequence.append((self._other_id, 'lr', self._allow_read))
        if self._allow_read:
            return self._other.lock_read()
        raise TestPreventLocking('lock_read disabled')

    def lock_write(self, token=None):
        self._sequence.append((self._other_id, 'lw', self._allow_write))
        if self._allow_write:
            return self._other.lock_write()
        raise TestPreventLocking('lock_write disabled')

    @only_raises(errors.LockNotHeld, errors.LockBroken)
    def unlock(self):
        self._sequence.append((self._other_id, 'ul', self._allow_unlock))
        if self._allow_unlock:
            return self._other.unlock()
        raise TestPreventLocking('unlock disabled')

    def disable_lock_read(self):
        """Make a lock_read call fail"""
        self.__dict__['_allow_read'] = False

    def disable_unlock(self):
        """Make an unlock call fail"""
        self.__dict__['_allow_unlock'] = False

    def disable_lock_write(self):
        """Make a lock_write call fail"""
        self.__dict__['_allow_write'] = False