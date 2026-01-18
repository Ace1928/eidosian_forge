from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def _build_hierarchy(self, dependencies, tree=None):
    tree = dict(top=True) if tree is None else tree
    for dep in dependencies:
        if dep.resource_name not in tree:
            tree[dep.resource_name] = dict(dep=dep, children=dict())
        if isinstance(dep, self.rm_models.Dependency) and dep.depends_on is not None and (len(dep.depends_on) > 0):
            self._build_hierarchy(dep.depends_on, tree[dep.resource_name]['children'])
    if 'top' in tree:
        tree.pop('top', None)
        keys = list(tree.keys())
        for key1 in keys:
            for key2 in keys:
                if key2 in tree and key1 in tree[key2]['children'] and (key1 in tree):
                    tree[key2]['children'][key1] = tree[key1]
                    tree.pop(key1)
    return tree