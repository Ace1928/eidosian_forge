from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddOrderByToListRequest(unused_ref, args, list_request):
    del unused_ref
    if args.sort_by:
        sort_by = [field.replace('~', '-') for field in args.sort_by]
        list_request.orderBy = ','.join(sort_by)
    return list_request