from ... import commands, config, errors, lazy_import, option, osutils
import stat
from breezy import (
def _force_clear(self, relpath):
    try:
        st = self._up_stat(relpath)
        if stat.S_ISDIR(st.st_mode):
            if not self.quiet:
                self.outf.write('Clearing {}/{}\n'.format(self.to_transport.external_url(), relpath))
            self._up_delete_tree(relpath)
        elif stat.S_ISLNK(st.st_mode):
            if not self.quiet:
                self.outf.write('Clearing {}/{}\n'.format(self.to_transport.external_url(), relpath))
            self._up_delete(relpath)
    except errors.PathError:
        pass