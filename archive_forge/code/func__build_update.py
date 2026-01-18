def _build_update(self, resource, attributes, updateable_attributes, non_updateable_attributes, **kwargs):
    update = {}
    resource = self.get_function(resource['id'])
    comparison_attributes = set(updateable_attributes if updateable_attributes is not None else attributes.keys()) - set(non_updateable_attributes if non_updateable_attributes is not None else [])
    resource_attributes = dict(((k, attributes[k]) for k in comparison_attributes if not self._is_equal(attributes[k], resource[k])))
    if resource_attributes:
        update['resource_attributes'] = resource_attributes
    return update