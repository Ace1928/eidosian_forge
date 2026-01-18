from keystoneauth1 import access
from keystoneauth1.identity import v3
from keystoneauth1.identity.v3 import federation
def _dependency_check():
    if requests_kerberos is None:
        raise ImportError('\nUsing the kerberos authentication plugin requires installation of additional\npackages. These can be installed with::\n\n    $ pip install keystoneauth1[kerberos]\n')