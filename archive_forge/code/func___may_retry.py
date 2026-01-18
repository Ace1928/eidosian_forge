import os
import ovs.util
import ovs.vlog
def __may_retry(self):
    if self.max_tries is None:
        return True
    elif self.max_tries > 0:
        self.max_tries -= 1
        return True
    else:
        return False