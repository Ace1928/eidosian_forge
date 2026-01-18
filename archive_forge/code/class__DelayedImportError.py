import inspect
import warnings
from traitlets import TraitType, TraitError, Undefined, Sentinel
class _DelayedImportError(object):

    def __init__(self, package_name):
        self.package_name = package_name

    def __getattribute__(self, name):
        package_name = super(_DelayedImportError, self).__getattribute__('package_name')
        raise RuntimeError('Missing dependency: %s' % package_name)