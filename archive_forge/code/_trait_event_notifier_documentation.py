from functools import partial
import types
import weakref
from traits.observation.exception_handling import handle_exception
from traits.observation.exceptions import NotifierNotFound
 Return true if the other notifier is equivalent to this one.

        Parameters
        ----------
        other : any

        Returns
        -------
        boolean
        