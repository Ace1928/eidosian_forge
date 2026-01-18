from celery import _state
from celery._state import app_or_default, disable_trace, enable_trace, pop_current_task, push_current_task
from celery.local import Proxy
from .base import Celery
from .utils import AppPickler
def __inner(fun):
    name = options.get('name')
    _state.connect_on_app_finalize(lambda app: app._task_from_fun(fun, **options))
    for app in _state._get_active_apps():
        if app.finalized:
            with app._finalize_mutex:
                app._task_from_fun(fun, **options)

    def task_by_cons():
        app = _state.get_current_app()
        return app.tasks[name or app.gen_task_name(fun.__name__, fun.__module__)]
    return Proxy(task_by_cons)