import logging
from oslo_vmware._i18n import _
def _print_deprecation_warning(clazz):
    LOG.warning('Exception %s is deprecated, it will be removed in the next release.', clazz.__name__)