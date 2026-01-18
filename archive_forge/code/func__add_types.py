import re
def _add_types(pool):
    """Add pool type formatting.

  Args:
    pool: the serializable storage pool
  Returns:
    nothing, it changes the input value.
  """
    types = '{}/{}/{}'.format(_format_pool_type(pool), _format_capacity_provisioning_type(pool), _format_perf_provisioning_type(pool))
    pool['formattedTypes'] = types