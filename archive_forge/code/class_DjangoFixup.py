import os
import sys
import warnings
from datetime import datetime, timezone
from importlib import import_module
from typing import IO, TYPE_CHECKING, Any, List, Optional, cast
from kombu.utils.imports import symbol_by_name
from kombu.utils.objects import cached_property
from celery import _state, signals
from celery.exceptions import FixupWarning, ImproperlyConfigured
class DjangoFixup:
    """Fixup installed when using Django."""

    def __init__(self, app: 'Celery'):
        self.app = app
        if _state.default_app is None:
            self.app.set_default()
        self._worker_fixup: Optional['DjangoWorkerFixup'] = None

    def install(self) -> 'DjangoFixup':
        sys.path.insert(0, os.getcwd())
        self._settings = symbol_by_name('django.conf:settings')
        self.app.loader.now = self.now
        if not self.app._custom_task_cls_used:
            self.app.task_cls = 'celery.contrib.django.task:DjangoTask'
        signals.import_modules.connect(self.on_import_modules)
        signals.worker_init.connect(self.on_worker_init)
        return self

    @property
    def worker_fixup(self) -> 'DjangoWorkerFixup':
        if self._worker_fixup is None:
            self._worker_fixup = DjangoWorkerFixup(self.app)
        return self._worker_fixup

    @worker_fixup.setter
    def worker_fixup(self, value: 'DjangoWorkerFixup') -> None:
        self._worker_fixup = value

    def on_import_modules(self, **kwargs: Any) -> None:
        self.worker_fixup.validate_models()

    def on_worker_init(self, **kwargs: Any) -> None:
        self.worker_fixup.install()

    def now(self, utc: bool=False) -> datetime:
        return datetime.now(timezone.utc) if utc else self._now()

    def autodiscover_tasks(self) -> List[str]:
        from django.apps import apps
        return [config.name for config in apps.get_app_configs()]

    @cached_property
    def _now(self) -> datetime:
        return symbol_by_name('django.utils.timezone:now')