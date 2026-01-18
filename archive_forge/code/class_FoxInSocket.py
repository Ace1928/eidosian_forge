from neutronclient._i18n import _
from neutronclient.common import extension
class FoxInSocket(extension.NeutronClientExtension):
    """Define required variables for resource operations."""
    resource = 'fox_socket'
    resource_plural = '%ss' % resource
    object_path = '/%s' % resource_plural
    resource_path = '/%s/%%s' % resource_plural
    versions = ['2.0']