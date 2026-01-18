from lxml import etree
import urllib
from .search import SearchManager
from .users import Users
from .resources import Project
from .tags import Tags
from .jsonutil import JsonTable
class GlobalManager(object):
    """ Mainly a container class to provide a clean interface for all
        management classes.
    """

    def __init__(self, interface):
        self._intf = interface
        self.search = SearchManager(self._intf)
        self.users = Users(self._intf)
        self.tags = Tags(self._intf)
        self.schemas = SchemaManager(self._intf)
        self.prearchive = PreArchive(self._intf)

    def register_callback(self, func):
        """ Defines a callback to execute when collections of resources are
            accessed.

            Parameters
            ----------
            func: callable
                A callable that takes the current collection object as first
                argument and the current element object as second argument.

            Examples
            --------
            >>> def notify(cobj, eobj):
            >>>    print eobj._uri
            >>> interface.manage.register_callback(notify)
        """
        self._intf._callback = func

    def unregister_callback(self):
        """ Unregisters the callback.
        """
        self._intf._callback = None

    def project(self, project_id):
        """ Returns a project manager.
        """
        return ProjectManager(project_id, self._intf)