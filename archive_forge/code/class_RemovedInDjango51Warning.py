import inspect
import warnings
from asgiref.sync import iscoroutinefunction, markcoroutinefunction, sync_to_async
class RemovedInDjango51Warning(DeprecationWarning):
    pass