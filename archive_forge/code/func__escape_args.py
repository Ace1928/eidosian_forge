import re
import warnings
from . import err
def _escape_args(self, args, conn):
    if isinstance(args, (tuple, list)):
        return tuple((conn.literal(arg) for arg in args))
    elif isinstance(args, dict):
        return {key: conn.literal(val) for key, val in args.items()}
    else:
        return conn.escape(args)