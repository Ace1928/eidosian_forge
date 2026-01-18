import importlib.resources as resources
def _in_proc_script_path():
    return resources.path(__package__, '_in_process.py')