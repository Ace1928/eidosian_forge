import re
def _maybe_add_throughput(pool):
    """Add throughput formatting.

  Args:
    pool: the serializable storage pool
  Returns:
    nothing, it changes the input value.
  """
    if not pool.get('poolProvisionedThroughput'):
        return
    provisioned_throughput = int(pool['poolProvisionedThroughput'])
    used_throughput = int(pool['status']['poolUsedThroughput'])
    formatted_throughput = '{:,}/{:,} (%{:.1f})'.format(used_throughput, provisioned_throughput, 100 * (used_throughput / provisioned_throughput))
    pool['formattedThroughput'] = formatted_throughput