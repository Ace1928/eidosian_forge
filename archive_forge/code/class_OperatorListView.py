import logging
from os_ken.services.protocols.bgp.operator.views import fields
class OperatorListView(OperatorAbstractView):

    def __init__(self, obj, filter_func=None):
        assert isinstance(obj, list)
        obj = RdyToFlattenList(obj)
        super(OperatorListView, self).__init__(obj, filter_func)

    def combine_related(self, field_name):
        f = self._fields[field_name]
        return CombinedViewsWrapper(RdyToFlattenList([f.retrieve_and_wrap(obj) for obj in self.model]))

    def get_field(self, field_name):
        f = self._fields[field_name]
        return RdyToFlattenList([f.get(obj) for obj in self.model])

    def encode(self):
        encoded_list = []
        for obj in self.model:
            encoded_item = {}
            for field_name, field in self._fields.items():
                if isinstance(field, fields.DataField):
                    encoded_item[field_name] = field.get(obj)
            encoded_list.append(encoded_item)
        return RdyToFlattenList(encoded_list)

    @property
    def model(self):
        if self._filter_func is not None:
            return RdyToFlattenList(filter(self._filter_func, self._obj))
        else:
            return self._obj