from oslo_config import cfg
def _scheduler_hints(self, scheduler_hints):
    """Augment scheduler hints with supplemental content."""
    if cfg.CONF.stack_scheduler_hints:
        if scheduler_hints is None:
            scheduler_hints = {}
        stack = self.stack
        scheduler_hints[self.HEAT_ROOT_STACK_ID] = stack.root_stack_id()
        scheduler_hints[self.HEAT_STACK_ID] = stack.id
        scheduler_hints[self.HEAT_STACK_NAME] = stack.name
        scheduler_hints[self.HEAT_PATH_IN_STACK] = self._path_in_stack(stack)
        scheduler_hints[self.HEAT_RESOURCE_NAME] = self.name
        scheduler_hints[self.HEAT_RESOURCE_UUID] = self.uuid
    return scheduler_hints