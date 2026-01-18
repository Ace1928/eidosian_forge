import os
import sysconfig
def _parse_python_tzpath(env_var):
    if not env_var:
        return ()
    raw_tzpath = env_var.split(os.pathsep)
    new_tzpath = tuple(filter(os.path.isabs, raw_tzpath))
    if len(new_tzpath) != len(raw_tzpath):
        import warnings
        msg = _get_invalid_paths_message(raw_tzpath)
        warnings.warn('Invalid paths specified in PYTHONTZPATH environment variable. ' + msg, InvalidTZPathWarning)
    return new_tzpath