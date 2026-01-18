import logging
from os_ken.services.protocols.bgp.operator.views import fields
class OperatorDictView(OperatorAbstractView):

    def __init__(self, obj, filter_func=None):
        assert isinstance(obj, dict)
        obj = RdyToFlattenDict(obj)
        super(OperatorDictView, self).__init__(obj, filter_func)

    def combine_related(self, field_name):
        f = self._fields[field_name]
        return CombinedViewsWrapper(RdyToFlattenList([f.retrieve_and_wrap(obj) for obj in self.model.values()]))

    def get_field(self, field_name):
        f = self._fields[field_name]
        dict_to_flatten = {}
        for key, obj in self.model.items():
            dict_to_flatten[key] = f.get(obj)
        return RdyToFlattenDict(dict_to_flatten)

    def encode(self):
        outer_dict_to_flatten = {}
        for key, obj in self.model.items():
            inner_dict_to_flatten = {}
            for field_name, field in self._fields.items():
                if isinstance(field, fields.DataField):
                    inner_dict_to_flatten[field_name] = field.get(obj)
            outer_dict_to_flatten[key] = inner_dict_to_flatten
        return RdyToFlattenDict(outer_dict_to_flatten)

    @property
    def model(self):
        if self._filter_func is not None:
            new_model = RdyToFlattenDict()
            for k, v in self._obj.items():
                if self._filter_func(k, v):
                    new_model[k] = v
            return new_model
        else:
            return self._obj