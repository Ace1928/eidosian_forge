from __future__ import print_function
import os,stat,time
import errno
import sys
class PickleShareLink:
    """ A shortdand for accessing nested PickleShare data conveniently.

    Created through PickleShareDB.getlink(), example::

        lnk = db.getlink('myobjects/test')
        lnk.foo = 2
        lnk.bar = lnk.foo + 5

    """

    def __init__(self, db, keydir):
        self.__dict__.update(locals())

    def __getattr__(self, key):
        return self.__dict__['db'][self.__dict__['keydir'] + '/' + key]

    def __setattr__(self, key, val):
        self.db[self.keydir + '/' + key] = val

    def __repr__(self):
        db = self.__dict__['db']
        keys = db.keys(self.__dict__['keydir'] + '/*')
        return "<PickleShareLink '%s': %s>" % (self.__dict__['keydir'], ';'.join([Path(k).basename() for k in keys]))