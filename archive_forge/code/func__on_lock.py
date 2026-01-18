import logging
import os
def _on_lock(self):
    content = self._content_template.format(pid=os.getpid(), hostname=LazyHostName())
    self._fp.write(' %s\n' % content)
    self._fp.truncate()