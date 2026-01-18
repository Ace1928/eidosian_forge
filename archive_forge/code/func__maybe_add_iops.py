import re
def _maybe_add_iops(pool):
    """Add iops formatting.

  Args:
    pool: the serializable storage pool
  Returns:
    nothing, it changes the input value.
  """
    if not pool.get('poolProvisionedIops'):
        return
    provisioned_iops = int(pool['poolProvisionedIops'])
    used_iops = int(pool['status']['poolUsedIops'])
    formatted_iops = '{:,}/{:,} ({:.1f}%)'.format(used_iops, provisioned_iops, 100 * (used_iops / provisioned_iops))
    pool['formattedIops'] = formatted_iops