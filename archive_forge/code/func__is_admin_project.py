import itertools
from oslo_serialization import jsonutils
import webob
def _is_admin_project(auth_ref):
    """Return an appropriate header value for X-Is-Admin-Project.

    Headers must be strings so we can't simply pass a boolean value through so
    return a True or False string to signal the admin project.
    """
    return 'True' if auth_ref.is_admin_project else 'False'