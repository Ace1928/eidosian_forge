from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def Hook(unused_ref, args, request):
    if not args.page_size:
        args.page_size = int(default_page_size)
    return request