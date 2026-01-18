import copy
import copyreg as copy_reg
import inspect
import pickle
import types
from io import StringIO as _cStringIO
from typing import Dict
from twisted.python import log, reflect
from twisted.python.compat import _PYPY
class Versioned:
    """
    This type of object is persisted with versioning information.

    I have a single class attribute, the int persistenceVersion.  After I am
    unserialized (and styles.doUpgrade() is called), self.upgradeToVersionX()
    will be called for each version upgrade I must undergo.

    For example, if I serialize an instance of a Foo(Versioned) at version 4
    and then unserialize it when the code is at version 9, the calls::

      self.upgradeToVersion5()
      self.upgradeToVersion6()
      self.upgradeToVersion7()
      self.upgradeToVersion8()
      self.upgradeToVersion9()

    will be made.  If any of these methods are undefined, a warning message
    will be printed.
    """
    persistenceVersion = 0
    persistenceForgets = ()

    def __setstate__(self, state):
        versionedsToUpgrade[id(self)] = self
        self.__dict__ = state

    def __getstate__(self, dict=None):
        """Get state, adding a version number to it on its way out."""
        dct = copy.copy(dict or self.__dict__)
        bases = _aybabtu(self.__class__)
        bases.reverse()
        bases.append(self.__class__)
        for base in bases:
            if 'persistenceForgets' in base.__dict__:
                for slot in base.persistenceForgets:
                    if slot in dct:
                        del dct[slot]
            if 'persistenceVersion' in base.__dict__:
                dct[f'{reflect.qual(base)}.persistenceVersion'] = base.persistenceVersion
        return dct

    def versionUpgrade(self):
        """(internal) Do a version upgrade."""
        bases = _aybabtu(self.__class__)
        bases.reverse()
        bases.append(self.__class__)
        if 'persistenceVersion' in self.__dict__:
            pver = self.__dict__['persistenceVersion']
            del self.__dict__['persistenceVersion']
            highestVersion = 0
            highestBase = None
            for base in bases:
                if 'persistenceVersion' not in base.__dict__:
                    continue
                if base.persistenceVersion > highestVersion:
                    highestBase = base
                    highestVersion = base.persistenceVersion
            if highestBase:
                self.__dict__['%s.persistenceVersion' % reflect.qual(highestBase)] = pver
        for base in bases:
            if Versioned not in base.__bases__ and 'persistenceVersion' not in base.__dict__:
                continue
            currentVers = base.persistenceVersion
            pverName = '%s.persistenceVersion' % reflect.qual(base)
            persistVers = self.__dict__.get(pverName) or 0
            if persistVers:
                del self.__dict__[pverName]
            assert persistVers <= currentVers, "Sorry, can't go backwards in time."
            while persistVers < currentVers:
                persistVers = persistVers + 1
                method = base.__dict__.get('upgradeToVersion%s' % persistVers, None)
                if method:
                    log.msg('Upgrading %s (of %s @ %s) to version %s' % (reflect.qual(base), reflect.qual(self.__class__), id(self), persistVers))
                    method(self)
                else:
                    log.msg('Warning: cannot upgrade {} to version {}'.format(base, persistVers))