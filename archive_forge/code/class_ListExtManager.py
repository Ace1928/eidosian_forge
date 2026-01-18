from cinderclient import base
from cinderclient import shell_utils
class ListExtManager(base.Manager):
    resource_class = ListExtResource

    def show_all(self):
        return self._list('/extensions', 'extensions')