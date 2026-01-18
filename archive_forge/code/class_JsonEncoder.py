import builtins
import json
import numbers
import operator
class JsonEncoder(json.JSONEncoder):
    """Customizable JSON encoder.

    If the object implements __getstate__, then that method is invoked, and its
    result is serialized instead of the object itself.
    """

    def default(self, value):
        try:
            get_state = value.__getstate__
        except AttributeError:
            pass
        else:
            return get_state()
        return super().default(value)