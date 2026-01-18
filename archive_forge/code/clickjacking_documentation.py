from functools import wraps
from asgiref.sync import iscoroutinefunction

    Modify a view function by setting a response variable that instructs
    XFrameOptionsMiddleware to NOT set the X-Frame-Options HTTP header. Usage:

    @xframe_options_exempt
    def some_view(request):
        ...
    