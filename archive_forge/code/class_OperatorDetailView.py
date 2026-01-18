import logging
from os_ken.services.protocols.bgp.operator.views import fields
class OperatorDetailView(OperatorAbstractView):

    def combine_related(self, field_name):
        f = self._fields[field_name]
        return CombinedViewsWrapper([f.retrieve_and_wrap(self._obj)])

    def get_field(self, field_name):
        f = self._fields[field_name]
        return f.get(self._obj)

    def encode(self):
        encoded = {}
        for field_name, field in self._fields.items():
            if isinstance(field, fields.DataField):
                encoded[field_name] = field.get(self._obj)
        return encoded

    def rel(self, field_name):
        f = self._fields[field_name]
        return f.retrieve_and_wrap(self._obj)

    @property
    def model(self):
        return self._obj