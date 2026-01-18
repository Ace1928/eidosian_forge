import re
import sys
import six
from six.moves.urllib import parse as urlparse
from routes.util import _url_quote as url_quote, _str_encode, as_unicode
def _defaults(self, routekeys, reserved_keys, kargs):
    """Creates default set with values stringified

        Put together our list of defaults, stringify non-None values
        and add in our action/id default if they use it and didn't
        specify it.

        defaultkeys is a list of the currently assumed default keys
        routekeys is a list of the keys found in the route path
        reserved_keys is a list of keys that are not

        """
    defaults = {}
    if 'controller' not in routekeys and 'controller' not in kargs and (not self.explicit):
        kargs['controller'] = 'content'
    if 'action' not in routekeys and 'action' not in kargs and (not self.explicit):
        kargs['action'] = 'index'
    defaultkeys = frozenset((key for key in kargs.keys() if key not in reserved_keys))
    for key in defaultkeys:
        if kargs[key] is not None:
            defaults[key] = self.make_unicode(kargs[key])
        else:
            defaults[key] = None
    if 'action' in routekeys and 'action' not in defaults and (not self.explicit):
        defaults['action'] = 'index'
    if 'id' in routekeys and 'id' not in defaults and (not self.explicit):
        defaults['id'] = None
    newdefaultkeys = frozenset((key for key in defaults.keys() if key not in reserved_keys))
    return (defaults, newdefaultkeys)