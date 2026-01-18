from oslo_log import log as logging
from heat.common import exception
from heat.engine import dependencies
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.objects import resource as resource_objects
class StackUpdate(object):
    """A Task to perform the update of an existing stack to a new template."""

    def __init__(self, existing_stack, new_stack, previous_stack, rollback=False):
        """Initialise with the existing stack and the new stack."""
        self.existing_stack = existing_stack
        self.new_stack = new_stack
        self.previous_stack = previous_stack
        self.rollback = rollback
        self.existing_snippets = dict(((n, r.frozen_definition()) for n, r in self.existing_stack.items() if n in self.new_stack))

    def __repr__(self):
        if self.rollback:
            return '%s Rollback' % str(self.existing_stack)
        else:
            return '%s Update' % str(self.existing_stack)

    def __call__(self):
        """Return a co-routine that updates the stack."""
        cleanup_prev = scheduler.DependencyTaskGroup(self.previous_stack.dependencies, self._remove_backup_resource, reverse=True)

        def get_error_wait_time(resource):
            return resource.cancel_grace_period()
        updater = scheduler.DependencyTaskGroup(self.dependencies(), self._resource_update, error_wait_time=get_error_wait_time)
        if not self.rollback:
            yield from cleanup_prev()
        try:
            yield from updater()
        finally:
            self.previous_stack.reset_dependencies()

    def _resource_update(self, res):
        if res.name in self.new_stack and self.new_stack[res.name] is res:
            return self._process_new_resource_update(res)
        else:
            return self._process_existing_resource_update(res)

    def _remove_backup_resource(self, prev_res):
        if prev_res.state not in ((prev_res.INIT, prev_res.COMPLETE), (prev_res.DELETE, prev_res.COMPLETE)):
            LOG.debug('Deleting backup resource %s', prev_res.name)
            yield from prev_res.destroy()

    @staticmethod
    def _exchange_stacks(existing_res, prev_res):
        resource_objects.Resource.exchange_stacks(existing_res.stack.context, existing_res.id, prev_res.id)
        prev_stack, existing_stack = (prev_res.stack, existing_res.stack)
        prev_stack.add_resource(existing_res)
        existing_stack.add_resource(prev_res)

    def _create_resource(self, new_res):
        res_name = new_res.name
        if res_name in self.previous_stack:
            prev_res = self.previous_stack[res_name]
            if prev_res.state not in ((prev_res.INIT, prev_res.COMPLETE), (prev_res.DELETE, prev_res.COMPLETE)):
                if prev_res.status == prev_res.COMPLETE:
                    LOG.debug('Swapping in backup Resource %s', res_name)
                    self._exchange_stacks(self.existing_stack[res_name], prev_res)
                    return
                LOG.debug('Deleting backup Resource %s', res_name)
                yield from prev_res.destroy()
        if res_name in self.existing_stack:
            LOG.debug('Backing up existing Resource %s', res_name)
            existing_res = self.existing_stack[res_name]
            self.previous_stack.add_resource(existing_res)
            existing_res.state_set(existing_res.UPDATE, existing_res.COMPLETE)
        self.existing_stack.add_resource(new_res)
        if res_name not in self.previous_stack.t[self.previous_stack.t.RESOURCES]:
            LOG.debug('Storing definition of new Resource %s', res_name)
            self.previous_stack.t.add_resource(new_res.t)
            self.previous_stack.t.store(self.previous_stack.context)
        yield from new_res.create()
        self._update_resource_data(new_res)

    def _check_replace_restricted(self, res):
        registry = res.stack.env.registry
        restricted_actions = registry.get_rsrc_restricted_actions(res.name)
        existing_res = self.existing_stack[res.name]
        if 'replace' in restricted_actions:
            ex = exception.ResourceActionRestricted(action='replace')
            failure = exception.ResourceFailure(ex, existing_res, existing_res.UPDATE)
            existing_res._add_event(existing_res.UPDATE, existing_res.FAILED, str(ex))
            raise failure

    def _update_resource_data(self, resource):
        node_data = resource.node_data(self.new_stack.defn)
        stk_defn.update_resource_data(self.existing_stack.defn, resource.name, node_data)
        stk_defn.update_resource_data(self.new_stack.defn, resource.name, node_data)

    def _process_new_resource_update(self, new_res):
        res_name = new_res.name
        if res_name in self.existing_stack:
            existing_res = self.existing_stack[res_name]
            is_substituted = existing_res.check_is_substituted(type(new_res))
            if type(existing_res) is type(new_res) or is_substituted:
                try:
                    yield from self._update_in_place(existing_res, new_res, is_substituted)
                except resource.UpdateReplace:
                    pass
                else:
                    LOG.debug('Storing definition of updated Resource %s', res_name)
                    self.previous_stack.t.add_resource(new_res.t)
                    self.previous_stack.t.store(self.previous_stack.context)
                    self.existing_stack.t.add_resource(new_res.t)
                    self.existing_stack.t.store(self.existing_stack.context)
                    LOG.info('Resource %(res_name)s for stack %(stack_name)s updated', {'res_name': res_name, 'stack_name': self.existing_stack.name})
                    self._update_resource_data(existing_res)
                    return
            else:
                self._check_replace_restricted(new_res)
        yield from self._create_resource(new_res)

    def _update_in_place(self, existing_res, new_res, is_substituted=False):
        existing_snippet = self.existing_snippets[existing_res.name]
        prev_res = self.previous_stack.get(new_res.name)
        new_snippet = new_res.t.reparse(self.existing_stack.defn, self.new_stack.t)
        if is_substituted:
            substitute = type(new_res)(existing_res.name, existing_res.t, existing_res.stack)
            existing_res.stack.resources[existing_res.name] = substitute
            existing_res = substitute
        existing_res.converge = self.new_stack.converge
        yield from existing_res.update(new_snippet, existing_snippet, prev_resource=prev_res)

    def _process_existing_resource_update(self, existing_res):
        res_name = existing_res.name
        if res_name in self.previous_stack:
            backup_res = self.previous_stack[res_name]
            yield from self._remove_backup_resource(backup_res)
        if res_name in self.new_stack:
            new_res = self.new_stack[res_name]
            if new_res.state == (new_res.INIT, new_res.COMPLETE):
                return
        if existing_res.stack is not self.previous_stack:
            yield from existing_res.destroy()
        if res_name not in self.new_stack:
            self.existing_stack.remove_resource(res_name)

    def dependencies(self):
        """Return the Dependencies graph for the update.

        Returns a Dependencies object representing the dependencies between
        update operations to move from an existing stack definition to a new
        one.
        """
        existing_deps = self.existing_stack.dependencies
        new_deps = self.new_stack.dependencies

        def edges():
            for e in new_deps.graph().edges():
                yield e
            for e in existing_deps.graph(reverse=True).edges():
                yield e
            for name, res in self.existing_stack.items():
                if name in self.new_stack:
                    yield (res, self.new_stack[name])
        return dependencies.Dependencies(edges())

    def preview(self):
        upd_keys = set(self.new_stack.resources.keys())
        cur_keys = set(self.existing_stack.resources.keys())
        common_keys = cur_keys.intersection(upd_keys)
        deleted_keys = cur_keys.difference(upd_keys)
        added_keys = upd_keys.difference(cur_keys)
        updated_keys = []
        replaced_keys = []
        for key in common_keys:
            current_res = self.existing_stack.resources[key]
            updated_res = self.new_stack.resources[key]
            current_props = current_res.frozen_definition().properties(current_res.properties_schema, current_res.context)
            updated_props = updated_res.frozen_definition().properties(updated_res.properties_schema, updated_res.context)
            if type(current_res) is not type(updated_res):
                replaced_keys.append(key)
                continue
            try:
                if current_res.preview_update(updated_res.frozen_definition(), current_res.frozen_definition(), updated_props, current_props, None):
                    updated_keys.append(key)
            except resource.UpdateReplace:
                replaced_keys.append(key)
        return {'unchanged': list(set(common_keys).difference(set(updated_keys + replaced_keys))), 'updated': updated_keys, 'replaced': replaced_keys, 'added': list(added_keys), 'deleted': list(deleted_keys)}