import importlib
import os
import time
import traceback
import warnings
from pathlib import Path
import django
from django.conf import global_settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.deprecation import RemovedInDjango51Warning, RemovedInDjango60Warning
from django.utils.functional import LazyObject, empty
class UserSettingsHolder:
    """Holder for user configured settings."""
    SETTINGS_MODULE = None

    def __init__(self, default_settings):
        """
        Requests for configuration variables not in this class are satisfied
        from the module specified in default_settings (if possible).
        """
        self.__dict__['_deleted'] = set()
        self.default_settings = default_settings

    def __getattr__(self, name):
        if not name.isupper() or name in self._deleted:
            raise AttributeError
        return getattr(self.default_settings, name)

    def __setattr__(self, name, value):
        self._deleted.discard(name)
        if name == 'DEFAULT_FILE_STORAGE':
            self.STORAGES[DEFAULT_STORAGE_ALIAS] = {'BACKEND': self.DEFAULT_FILE_STORAGE}
            warnings.warn(DEFAULT_FILE_STORAGE_DEPRECATED_MSG, RemovedInDjango51Warning)
        if name == 'STATICFILES_STORAGE':
            self.STORAGES[STATICFILES_STORAGE_ALIAS] = {'BACKEND': self.STATICFILES_STORAGE}
            warnings.warn(STATICFILES_STORAGE_DEPRECATED_MSG, RemovedInDjango51Warning)
        if name == 'FORMS_URLFIELD_ASSUME_HTTPS':
            warnings.warn(FORMS_URLFIELD_ASSUME_HTTPS_DEPRECATED_MSG, RemovedInDjango60Warning)
        super().__setattr__(name, value)
        if name == 'STORAGES':
            if (default_file_storage := self.STORAGES.get(DEFAULT_STORAGE_ALIAS)):
                super().__setattr__('DEFAULT_FILE_STORAGE', default_file_storage.get('BACKEND'))
            else:
                self.STORAGES.setdefault(DEFAULT_STORAGE_ALIAS, {'BACKEND': 'django.core.files.storage.FileSystemStorage'})
            if (staticfiles_storage := self.STORAGES.get(STATICFILES_STORAGE_ALIAS)):
                super().__setattr__('STATICFILES_STORAGE', staticfiles_storage.get('BACKEND'))
            else:
                self.STORAGES.setdefault(STATICFILES_STORAGE_ALIAS, {'BACKEND': 'django.contrib.staticfiles.storage.StaticFilesStorage'})

    def __delattr__(self, name):
        self._deleted.add(name)
        if hasattr(self, name):
            super().__delattr__(name)

    def __dir__(self):
        return sorted((s for s in [*self.__dict__, *dir(self.default_settings)] if s not in self._deleted))

    def is_overridden(self, setting):
        deleted = setting in self._deleted
        set_locally = setting in self.__dict__
        set_on_default = getattr(self.default_settings, 'is_overridden', lambda s: False)(setting)
        return deleted or set_locally or set_on_default

    def __repr__(self):
        return '<%(cls)s>' % {'cls': self.__class__.__name__}