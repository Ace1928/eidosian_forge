import re
def _format_capacity_provisioning_type(pool):
    """Format capacity provisioning type.

  Args:
    pool: the serializable storage pool
  Returns:
    the abbreviated string
  """
    return PROVISIONING_TYPE_ABBREVIATIONS.get(pool['capacityProvisioningType'], UNKNOWN_TYPE_PLACEHOLDER)