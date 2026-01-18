from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
class TagsModule(BaseModule):

    def build_entity(self):
        return otypes.Tag(id=self._module.params['id'], name=self._module.params['name'], description=self._module.params['description'], parent=otypes.Tag(name=self._module.params['parent']) if self._module.params['parent'] else None)

    def post_create(self, entity):
        self.update_check(entity)

    def _update_tag_assignments(self, entity, name):
        if self._module.params[name] is None:
            return
        state = self.param('state')
        entities_service = getattr(self._connection.system_service(), '%s_service' % name)()
        current_vms = [vm.name for vm in entities_service.list(search='tag=%s' % self._module.params['name'])]
        if state in ['present', 'attached', 'detached']:
            for entity_name in self._module.params[name]:
                entity_id = get_id_by_name(entities_service, entity_name)
                tags_service = entities_service.service(entity_id).tags_service()
                current_tags = [tag.name for tag in tags_service.list()]
                if state in ['attached', 'present']:
                    if self._module.params['name'] not in current_tags:
                        if not self._module.check_mode:
                            tags_service.add(tag=otypes.Tag(name=self._module.params['name']))
                        self.changed = True
                elif state == 'detached':
                    if self._module.params['name'] in current_tags:
                        tag_id = get_id_by_name(tags_service, self.param('name'))
                        if not self._module.check_mode:
                            tags_service.tag_service(tag_id).remove()
                        self.changed = True
        if state == 'present':
            for entity_name in [e for e in current_vms if e not in self._module.params[name]]:
                if not self._module.check_mode:
                    entity_id = get_id_by_name(entities_service, entity_name)
                    tags_service = entities_service.service(entity_id).tags_service()
                    tag_id = get_id_by_name(tags_service, self.param('name'))
                    tags_service.tag_service(tag_id).remove()
                self.changed = True

    def _get_parent(self, entity):
        parent = None
        if entity.parent:
            parent = self._connection.follow_link(entity.parent).name
        return parent

    def update_check(self, entity):
        self._update_tag_assignments(entity, 'vms')
        self._update_tag_assignments(entity, 'hosts')
        return equal(self._module.params.get('description'), entity.description) and equal(self._module.params.get('name'), entity.name) and equal(self._module.params.get('parent'), self._get_parent(entity))