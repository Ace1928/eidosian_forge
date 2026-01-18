from celery.utils.functional import firstmethod, mlazy
from celery.utils.imports import instantiate
class MapAnnotation(dict):
    """Annotation map: task_name => attributes."""

    def annotate_any(self):
        try:
            return dict(self['*'])
        except KeyError:
            pass

    def annotate(self, task):
        try:
            return dict(self[task.name])
        except KeyError:
            pass