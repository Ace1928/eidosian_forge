from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddOrderAllocationEntryArgs(parser):
    """Register an arg group for Order Allocation entry flags.

  Args:
    parser: A group where all allocation entry arguments are registered.

  Returns:
    No return value.
  """
    resource_value_group = parser.add_mutually_exclusive_group(required=True)
    resource_value_group.add_argument('--int64-resource-value', type=int, help='Resource value in int64 type.')
    resource_value_group.add_argument('--double-resource-value', type=float, help='Resource value in double type.')
    resource_value_group.add_argument('--string-resource-value', help='Resource value in string type.')
    parser.add_argument('--targets', required=True, action='append', help='Targets of the order allocation. Only projects are allowed now.')