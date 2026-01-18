from __future__ import annotations
import os
from streamlit import config, file_util, util
class FolderBlackList:
    """Implement a black list object with globbing.

    Note
    ----
    Blacklist any path that matches a glob in `DEFAULT_FOLDER_BLACKLIST`.

    """

    def __init__(self, folder_blacklist):
        """Constructor.

        Parameters
        ----------
        folder_blacklist : list of str
            list of folder names with globbing to blacklist.

        """
        self._folder_blacklist = list(folder_blacklist)
        self._folder_blacklist.extend(DEFAULT_FOLDER_BLACKLIST)
        if config.get_option('global.developmentMode'):
            self._folder_blacklist.append(os.path.dirname(__file__))

    def __repr__(self) -> str:
        return util.repr_(self)

    def is_blacklisted(self, filepath):
        """Test if filepath is in the blacklist.

        Parameters
        ----------
        filepath : str
            File path that we intend to test.

        """
        return any((file_util.file_is_in_folder_glob(filepath, blacklisted_folder) for blacklisted_folder in self._folder_blacklist))