import re
def _format_single(pool):
    """Format a single pool for displaying it in the list response."""
    _add_types(pool)
    _add_capacity(pool)
    _maybe_add_iops(pool)
    _maybe_add_throughput(pool)
    return pool