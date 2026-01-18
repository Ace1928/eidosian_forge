import os
from magnumclient.i18n import _
from osc_lib.command import command
def _show_cert(certificate):
    try:
        print(certificate.pem)
    except AttributeError:
        return None