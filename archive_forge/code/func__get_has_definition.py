import logging
from botocore import xform_name
def _get_has_definition(self):
    """
        Get a ``has`` relationship definition from a model, where the
        service resource model is treated special in that it contains
        a relationship to every resource defined for the service. This
        allows things like ``s3.Object('bucket-name', 'key')`` to
        work even though the JSON doesn't define it explicitly.

        :rtype: dict
        :return: Mapping of names to subresource and reference
                 definitions.
        """
    if self.name not in self._resource_defs:
        definition = {}
        for name, resource_def in self._resource_defs.items():
            found = False
            has_items = self._definition.get('has', {}).items()
            for has_name, has_def in has_items:
                if has_def.get('resource', {}).get('type') == name:
                    definition[has_name] = has_def
                    found = True
            if not found:
                fake_has = {'resource': {'type': name, 'identifiers': []}}
                for identifier in resource_def.get('identifiers', []):
                    fake_has['resource']['identifiers'].append({'target': identifier['name'], 'source': 'input'})
                definition[name] = fake_has
    else:
        definition = self._definition.get('has', {})
    return definition