import os
import time
import calendar
import socket
import errno
import copy
import warnings
import email
import email.message
import email.generator
import io
import contextlib
from types import GenericAlias
class MH(Mailbox):
    """An MH mailbox."""

    def __init__(self, path, factory=None, create=True):
        """Initialize an MH instance."""
        Mailbox.__init__(self, path, factory, create)
        if not os.path.exists(self._path):
            if create:
                os.mkdir(self._path, 448)
                os.close(os.open(os.path.join(self._path, '.mh_sequences'), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 384))
            else:
                raise NoSuchMailboxError(self._path)
        self._locked = False

    def add(self, message):
        """Add message and return assigned key."""
        keys = self.keys()
        if len(keys) == 0:
            new_key = 1
        else:
            new_key = max(keys) + 1
        new_path = os.path.join(self._path, str(new_key))
        f = _create_carefully(new_path)
        closed = False
        try:
            if self._locked:
                _lock_file(f)
            try:
                try:
                    self._dump_message(message, f)
                except BaseException:
                    if self._locked:
                        _unlock_file(f)
                    _sync_close(f)
                    closed = True
                    os.remove(new_path)
                    raise
                if isinstance(message, MHMessage):
                    self._dump_sequences(message, new_key)
            finally:
                if self._locked:
                    _unlock_file(f)
        finally:
            if not closed:
                _sync_close(f)
        return new_key

    def remove(self, key):
        """Remove the keyed message; raise KeyError if it doesn't exist."""
        path = os.path.join(self._path, str(key))
        try:
            f = open(path, 'rb+')
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise KeyError('No message with key: %s' % key)
            else:
                raise
        else:
            f.close()
            os.remove(path)

    def __setitem__(self, key, message):
        """Replace the keyed message; raise KeyError if it doesn't exist."""
        path = os.path.join(self._path, str(key))
        try:
            f = open(path, 'rb+')
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise KeyError('No message with key: %s' % key)
            else:
                raise
        try:
            if self._locked:
                _lock_file(f)
            try:
                os.close(os.open(path, os.O_WRONLY | os.O_TRUNC))
                self._dump_message(message, f)
                if isinstance(message, MHMessage):
                    self._dump_sequences(message, key)
            finally:
                if self._locked:
                    _unlock_file(f)
        finally:
            _sync_close(f)

    def get_message(self, key):
        """Return a Message representation or raise a KeyError."""
        try:
            if self._locked:
                f = open(os.path.join(self._path, str(key)), 'rb+')
            else:
                f = open(os.path.join(self._path, str(key)), 'rb')
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise KeyError('No message with key: %s' % key)
            else:
                raise
        with f:
            if self._locked:
                _lock_file(f)
            try:
                msg = MHMessage(f)
            finally:
                if self._locked:
                    _unlock_file(f)
        for name, key_list in self.get_sequences().items():
            if key in key_list:
                msg.add_sequence(name)
        return msg

    def get_bytes(self, key):
        """Return a bytes representation or raise a KeyError."""
        try:
            if self._locked:
                f = open(os.path.join(self._path, str(key)), 'rb+')
            else:
                f = open(os.path.join(self._path, str(key)), 'rb')
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise KeyError('No message with key: %s' % key)
            else:
                raise
        with f:
            if self._locked:
                _lock_file(f)
            try:
                return f.read().replace(linesep, b'\n')
            finally:
                if self._locked:
                    _unlock_file(f)

    def get_file(self, key):
        """Return a file-like representation or raise a KeyError."""
        try:
            f = open(os.path.join(self._path, str(key)), 'rb')
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise KeyError('No message with key: %s' % key)
            else:
                raise
        return _ProxyFile(f)

    def iterkeys(self):
        """Return an iterator over keys."""
        return iter(sorted((int(entry) for entry in os.listdir(self._path) if entry.isdigit())))

    def __contains__(self, key):
        """Return True if the keyed message exists, False otherwise."""
        return os.path.exists(os.path.join(self._path, str(key)))

    def __len__(self):
        """Return a count of messages in the mailbox."""
        return len(list(self.iterkeys()))

    def lock(self):
        """Lock the mailbox."""
        if not self._locked:
            self._file = open(os.path.join(self._path, '.mh_sequences'), 'rb+')
            _lock_file(self._file)
            self._locked = True

    def unlock(self):
        """Unlock the mailbox if it is locked."""
        if self._locked:
            _unlock_file(self._file)
            _sync_close(self._file)
            del self._file
            self._locked = False

    def flush(self):
        """Write any pending changes to the disk."""
        return

    def close(self):
        """Flush and close the mailbox."""
        if self._locked:
            self.unlock()

    def list_folders(self):
        """Return a list of folder names."""
        result = []
        for entry in os.listdir(self._path):
            if os.path.isdir(os.path.join(self._path, entry)):
                result.append(entry)
        return result

    def get_folder(self, folder):
        """Return an MH instance for the named folder."""
        return MH(os.path.join(self._path, folder), factory=self._factory, create=False)

    def add_folder(self, folder):
        """Create a folder and return an MH instance representing it."""
        return MH(os.path.join(self._path, folder), factory=self._factory)

    def remove_folder(self, folder):
        """Delete the named folder, which must be empty."""
        path = os.path.join(self._path, folder)
        entries = os.listdir(path)
        if entries == ['.mh_sequences']:
            os.remove(os.path.join(path, '.mh_sequences'))
        elif entries == []:
            pass
        else:
            raise NotEmptyError('Folder not empty: %s' % self._path)
        os.rmdir(path)

    def get_sequences(self):
        """Return a name-to-key-list dictionary to define each sequence."""
        results = {}
        with open(os.path.join(self._path, '.mh_sequences'), 'r', encoding='ASCII') as f:
            all_keys = set(self.keys())
            for line in f:
                try:
                    name, contents = line.split(':')
                    keys = set()
                    for spec in contents.split():
                        if spec.isdigit():
                            keys.add(int(spec))
                        else:
                            start, stop = (int(x) for x in spec.split('-'))
                            keys.update(range(start, stop + 1))
                    results[name] = [key for key in sorted(keys) if key in all_keys]
                    if len(results[name]) == 0:
                        del results[name]
                except ValueError:
                    raise FormatError('Invalid sequence specification: %s' % line.rstrip())
        return results

    def set_sequences(self, sequences):
        """Set sequences using the given name-to-key-list dictionary."""
        f = open(os.path.join(self._path, '.mh_sequences'), 'r+', encoding='ASCII')
        try:
            os.close(os.open(f.name, os.O_WRONLY | os.O_TRUNC))
            for name, keys in sequences.items():
                if len(keys) == 0:
                    continue
                f.write(name + ':')
                prev = None
                completing = False
                for key in sorted(set(keys)):
                    if key - 1 == prev:
                        if not completing:
                            completing = True
                            f.write('-')
                    elif completing:
                        completing = False
                        f.write('%s %s' % (prev, key))
                    else:
                        f.write(' %s' % key)
                    prev = key
                if completing:
                    f.write(str(prev) + '\n')
                else:
                    f.write('\n')
        finally:
            _sync_close(f)

    def pack(self):
        """Re-name messages to eliminate numbering gaps. Invalidates keys."""
        sequences = self.get_sequences()
        prev = 0
        changes = []
        for key in self.iterkeys():
            if key - 1 != prev:
                changes.append((key, prev + 1))
                try:
                    os.link(os.path.join(self._path, str(key)), os.path.join(self._path, str(prev + 1)))
                except (AttributeError, PermissionError):
                    os.rename(os.path.join(self._path, str(key)), os.path.join(self._path, str(prev + 1)))
                else:
                    os.unlink(os.path.join(self._path, str(key)))
            prev += 1
        self._next_key = prev + 1
        if len(changes) == 0:
            return
        for name, key_list in sequences.items():
            for old, new in changes:
                if old in key_list:
                    key_list[key_list.index(old)] = new
        self.set_sequences(sequences)

    def _dump_sequences(self, message, key):
        """Inspect a new MHMessage and update sequences appropriately."""
        pending_sequences = message.get_sequences()
        all_sequences = self.get_sequences()
        for name, key_list in all_sequences.items():
            if name in pending_sequences:
                key_list.append(key)
            elif key in key_list:
                del key_list[key_list.index(key)]
        for sequence in pending_sequences:
            if sequence not in all_sequences:
                all_sequences[sequence] = [key]
        self.set_sequences(all_sequences)