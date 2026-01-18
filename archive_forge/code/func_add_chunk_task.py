from celery._state import connect_on_app_finalize
from celery.utils.log import get_logger
@connect_on_app_finalize
def add_chunk_task(app):
    from celery.canvas import chunks as _chunks

    @app.task(name='celery.chunks', shared=False, lazy=False)
    def chunks(task, it, n):
        return _chunks.apply_chunks(task, it, n)
    return chunks