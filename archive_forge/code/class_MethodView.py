from __future__ import annotations
import typing as t
from . import typing as ft
from .globals import current_app
from .globals import request
class MethodView(View):
    """Dispatches request methods to the corresponding instance methods.
    For example, if you implement a ``get`` method, it will be used to
    handle ``GET`` requests.

    This can be useful for defining a REST API.

    :attr:`methods` is automatically set based on the methods defined on
    the class.

    See :doc:`views` for a detailed guide.

    .. code-block:: python

        class CounterAPI(MethodView):
            def get(self):
                return str(session.get("counter", 0))

            def post(self):
                session["counter"] = session.get("counter", 0) + 1
                return redirect(url_for("counter"))

        app.add_url_rule(
            "/counter", view_func=CounterAPI.as_view("counter")
        )
    """

    def __init_subclass__(cls, **kwargs: t.Any) -> None:
        super().__init_subclass__(**kwargs)
        if 'methods' not in cls.__dict__:
            methods = set()
            for base in cls.__bases__:
                if getattr(base, 'methods', None):
                    methods.update(base.methods)
            for key in http_method_funcs:
                if hasattr(cls, key):
                    methods.add(key.upper())
            if methods:
                cls.methods = methods

    def dispatch_request(self, **kwargs: t.Any) -> ft.ResponseReturnValue:
        meth = getattr(self, request.method.lower(), None)
        if meth is None and request.method == 'HEAD':
            meth = getattr(self, 'get', None)
        assert meth is not None, f'Unimplemented method {request.method!r}'
        return current_app.ensure_sync(meth)(**kwargs)