from celery._state import connect_on_app_finalize
from celery.utils.log import get_logger
@connect_on_app_finalize
def add_accumulate_task(app):
    """Task used by Task.replace when replacing task with group."""

    @app.task(bind=True, name='celery.accumulate', shared=False, lazy=False)
    def accumulate(self, *args, **kwargs):
        index = kwargs.get('index')
        return args[index] if index is not None else args
    return accumulate