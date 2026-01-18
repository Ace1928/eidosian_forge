import hashlib
import logging
import shelve
class EptidShelve(Eptid):

    def __init__(self, secret, filename):
        Eptid.__init__(self, secret)
        if filename.endswith('.db'):
            filename = filename.rsplit('.db', 1)[0]
        self._db = shelve.open(filename, writeback=True, protocol=2)

    def close(self):
        self._db.close()