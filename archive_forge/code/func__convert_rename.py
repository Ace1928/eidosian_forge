from .. import (
import stat
def _convert_rename(self, fc):
    """Convert a FileRenameCommand into a new FileCommand.

        :return: None if the rename is being ignored, otherwise a
          new FileCommand based on the whether the old and new paths
          are inside or outside of the interesting locations.
          """
    old = fc.old_path
    new = fc.new_path
    keep_old = self._path_to_be_kept(old)
    keep_new = self._path_to_be_kept(new)
    if keep_old and keep_new:
        fc.old_path = self._adjust_for_new_root(old)
        fc.new_path = self._adjust_for_new_root(new)
        return fc
    elif keep_old:
        old = self._adjust_for_new_root(old)
        return commands.FileDeleteCommand(old)
    elif keep_new:
        self.warning('cannot turn rename of %s into an add of %s yet' % (old, new))
    return None