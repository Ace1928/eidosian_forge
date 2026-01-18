from . import branch, controldir, errors, trace, ui, urlutils
from .i18n import gettext
def _select_bind_location(self):
    """Select a location to bind or create a reference to.

        Preference is:
        1. user specified location
        2. branch reference location (it's a kind of bind location)
        3. current bind location
        4. previous bind location (it was a good choice once)
        5. push location (it's writeable, so committable)
        6. parent location (it's pullable, so update-from-able)
        """
    if self.new_bound_location is not None:
        return self.new_bound_location
    if self.local_branch is not None:
        bound = self.local_branch.get_bound_location()
        if bound is not None:
            return bound
        old_bound = self.local_branch.get_old_bound_location()
        if old_bound is not None:
            return old_bound
        push_location = self.local_branch.get_push_location()
        if push_location is not None:
            return push_location
        parent = self.local_branch.get_parent()
        if parent is not None:
            return parent
    elif self.referenced_branch is not None:
        return self.referenced_branch.base
    raise NoBindLocation(self.controldir)