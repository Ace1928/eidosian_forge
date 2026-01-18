import os
import time
from twisted.internet import defer
from twisted.names import common, dns, error
from twisted.python import failure
from twisted.python.compat import execfile, nativeString
from twisted.python.filepath import FilePath
def _cbAllRecords(self, results):
    ans, auth, add = ([], [], [])
    for res in results:
        if res[0]:
            ans.extend(res[1][0])
            auth.extend(res[1][1])
            add.extend(res[1][2])
    return (ans, auth, add)