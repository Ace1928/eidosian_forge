from .. import (
import stat
def _filter_filecommands(self, filecmd_iter):
    """Return the filecommands filtered by includes & excludes.

        :return: a list of FileCommand objects
        """
    if self.includes is None and self.excludes is None:
        return list(filecmd_iter())
    result = []
    for fc in filecmd_iter():
        if isinstance(fc, commands.FileModifyCommand) or isinstance(fc, commands.FileDeleteCommand):
            if self._path_to_be_kept(fc.path):
                fc.path = self._adjust_for_new_root(fc.path)
            else:
                continue
        elif isinstance(fc, commands.FileDeleteAllCommand):
            pass
        elif isinstance(fc, commands.FileRenameCommand):
            fc = self._convert_rename(fc)
        elif isinstance(fc, commands.FileCopyCommand):
            fc = self._convert_copy(fc)
        else:
            self.warning('cannot handle FileCommands of class %s - ignoring', fc.__class__)
            continue
        if fc is not None:
            result.append(fc)
    return result