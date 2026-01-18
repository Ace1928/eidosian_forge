import importlib
import inspect
class OptionalDataField(DataField):

    def get(self, obj):
        if hasattr(obj, self.field_name):
            return getattr(obj, self.field_name)
        else:
            return None