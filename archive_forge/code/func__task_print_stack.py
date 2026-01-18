import linecache
import reprlib
import traceback
from . import base_futures
from . import coroutines
def _task_print_stack(task, limit, file):
    extracted_list = []
    checked = set()
    for f in task.get_stack(limit=limit):
        lineno = f.f_lineno
        co = f.f_code
        filename = co.co_filename
        name = co.co_name
        if filename not in checked:
            checked.add(filename)
            linecache.checkcache(filename)
        line = linecache.getline(filename, lineno, f.f_globals)
        extracted_list.append((filename, lineno, name, line))
    exc = task._exception
    if not extracted_list:
        print(f'No stack for {task!r}', file=file)
    elif exc is not None:
        print(f'Traceback for {task!r} (most recent call last):', file=file)
    else:
        print(f'Stack for {task!r} (most recent call last):', file=file)
    traceback.print_list(extracted_list, file=file)
    if exc is not None:
        for line in traceback.format_exception_only(exc.__class__, exc):
            print(line, file=file, end='')