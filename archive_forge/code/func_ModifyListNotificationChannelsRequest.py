from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def ModifyListNotificationChannelsRequest(project_ref, args, list_request):
    """Modifies the list request by adding a filter defined by the type flag."""
    del project_ref
    filter_expr = args.filter
    if args.type:
        filter_expr = _AddTypeToFilter(filter_expr, args.type)
    list_request.filter = filter_expr
    return list_request