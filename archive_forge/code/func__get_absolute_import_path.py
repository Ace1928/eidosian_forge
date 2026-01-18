import inspect
import pennylane as qml
def _get_absolute_import_path(fn):
    return f'{inspect.getmodule(fn).__name__}.{fn.__name__}'